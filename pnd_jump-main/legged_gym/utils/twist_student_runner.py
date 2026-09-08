"""TWIST-style runner: PPO on 58-dim student + annealed BC to future-aware teacher."""
from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCriticRecurrent


class TwistStudentRunner:
    STUDENT_OBS = 58
    STUDENT_PRIV = 61
    TEACHER_OBS = 82

    def __init__(
        self,
        env,
        train_cfg,
        teacher,
        log_dir=None,
        device="cpu",
        teacher_future_offsets=(5, 10),
    ):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = dict(train_cfg["algorithm"])
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.teacher = teacher
        self.teacher_future_offsets = tuple(teacher_future_offsets)

        self.bc_coef_init = float(self.alg_cfg.pop("bc_coef_init", 1.0))
        self.bc_coef_final = float(self.alg_cfg.pop("bc_coef_final", 0.05))
        self.bc_anneal_iters = int(self.alg_cfg.pop("bc_anneal_iters", 2000))

        student = ActorCriticRecurrent(
            self.STUDENT_OBS,
            self.STUDENT_PRIV,
            env.num_actions,
            **self.policy_cfg,
        ).to(device)
        self.alg = PPO(student, device=device, **self.alg_cfg)
        self.num_steps_per_env = int(self.cfg["num_steps_per_env"])
        self.save_interval = int(self.cfg["save_interval"])
        self.alg.init_storage(
            env.num_envs,
            self.num_steps_per_env,
            [self.STUDENT_OBS],
            [self.STUDENT_PRIV],
            [env.num_actions],
        )

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.current_learning_iteration = 0
        _, _ = self.env.reset()

    def _student_views(self, obs, priv):
        stu_obs = obs[:, : self.STUDENT_OBS]
        if priv is None:
            stu_priv = stu_obs
        else:
            # priv layout: lin_vel(3) + full obs
            stu_priv = torch.cat((priv[:, :3], obs[:, : self.STUDENT_OBS]), dim=-1)
        return stu_obs, stu_priv

    def _bc_coef(self, it):
        if self.bc_anneal_iters <= 0:
            return self.bc_coef_final
        t = min(max(it, 0), self.bc_anneal_iters) / float(self.bc_anneal_iters)
        return self.bc_coef_init * (1.0 - t) + self.bc_coef_final * t

    def _detach_hidden(self, ac):
        for mem in (ac.memory_a, ac.memory_c):
            if mem.hidden_states is None:
                continue
            mem.hidden_states = tuple(h.detach() for h in mem.hidden_states)

    def _bc_update(self, obs_buf, tea_mu_buf, dones_buf, bc_coef):
        """BPTT MSE on action means: student(obs_t) → teacher_mu_t."""
        if bc_coef <= 0:
            return 0.0
        student = self.alg.actor_critic
        student.train()
        # Rollout buffers were filled under inference_mode; clone for autograd.
        obs_buf = obs_buf.detach().clone()
        tea_mu_buf = tea_mu_buf.detach().clone()
        dones_buf = dones_buf.detach().clone()
        T, N, _ = obs_buf.shape
        # Drop inference-mode LSTM states left over from PPO rollout/update.
        student.memory_a.hidden_states = None
        student.memory_c.hidden_states = None
        # Prime fresh hidden states
        _ = student.act_inference(obs_buf[0])
        _ = student.evaluate(
            torch.cat(
                (torch.zeros(N, 3, device=self.device), obs_buf[0]),
                dim=-1,
            )
        )
        student.reset(torch.ones(N, dtype=torch.bool, device=self.device))

        loss = 0.0
        for t in range(T):
            pred = student.act_inference(obs_buf[t])
            loss = loss + F.mse_loss(pred, tea_mu_buf[t])
            done_t = dones_buf[t].to(dtype=torch.bool)
            if torch.any(done_t):
                student.reset(done_t)
                self._detach_hidden(student)
        loss = (loss / T) * bc_coef

        self.alg.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), self.alg.max_grad_norm)
        self.alg.optimizer.step()
        # Fresh LSTM state for the next PPO rollout.
        student.memory_a.hidden_states = None
        student.memory_c.hidden_states = None
        return float(loss.detach().item())

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        priv = self.env.get_privileged_observations()
        priv = priv.to(self.device) if priv is not None else None
        stu_obs, stu_priv = self._student_views(obs, priv)

        self.alg.actor_critic.train()
        self.teacher.eval()
        # Prime teacher actor+critic LSTM before reset (hidden starts as None).
        tea0 = (
            self.env.get_teacher_observations(self.teacher_future_offsets).to(self.device)
            if hasattr(self.env, "get_teacher_observations")
            else obs
        )
        _ = self.teacher.act_inference(tea0)
        tea_priv0 = torch.cat((priv[:, :3], tea0), dim=-1) if priv is not None else tea0
        _ = self.teacher.evaluate(tea_priv0)
        self.teacher.reset(torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device))

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        obs_buf = torch.zeros(
            self.num_steps_per_env,
            self.env.num_envs,
            self.STUDENT_OBS,
            device=self.device,
        )
        tea_mu_buf = torch.zeros(
            self.num_steps_per_env,
            self.env.num_envs,
            self.env.num_actions,
            device=self.device,
        )
        dones_buf = torch.zeros(
            self.num_steps_per_env, self.env.num_envs, device=self.device
        )

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            bc_coef = self._bc_coef(it - start_iter)
            start = time.time()
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(stu_obs, stu_priv)
                    if hasattr(self.env, "get_teacher_observations"):
                        tea_obs = self.env.get_teacher_observations(
                            self.teacher_future_offsets
                        ).to(self.device)
                    else:
                        tea_obs = obs
                    tea_mu = self.teacher.act_inference(tea_obs)

                    obs_buf[i].copy_(stu_obs)
                    tea_mu_buf[i].copy_(tea_mu)

                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    obs = obs.to(self.device)
                    priv = (
                        privileged_obs.to(self.device)
                        if privileged_obs is not None
                        else None
                    )
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    dones_buf[i].copy_(dones)
                    stu_obs, stu_priv = self._student_views(obs, priv)
                    self.alg.process_env_step(rewards, dones, infos)
                    if torch.any(dones):
                        self.teacher.reset(dones.to(dtype=torch.bool))

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                collection_time = time.time() - start
                start = time.time()
                self.alg.compute_returns(stu_priv)

            mean_value_loss, mean_surrogate_loss = self.alg.update()
            mean_bc_loss = self._bc_update(obs_buf, tea_mu_buf, dones_buf, bc_coef)
            learn_time = time.time() - start

            self.current_learning_iteration = it
            if self.log_dir is not None:
                self._log(
                    it,
                    start_iter,
                    tot_iter,
                    collection_time,
                    learn_time,
                    mean_value_loss,
                    mean_surrogate_loss,
                    mean_bc_loss,
                    bc_coef,
                    ep_infos,
                    rewbuffer,
                    lenbuffer,
                )
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()

        self.current_learning_iteration = tot_iter
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{tot_iter}.pt"))

    def _log(
        self,
        it,
        start_iter,
        tot_iter,
        collection_time,
        learn_time,
        mean_value_loss,
        mean_surrogate_loss,
        mean_bc_loss,
        bc_coef,
        ep_infos,
        rewbuffer,
        lenbuffer,
        width=80,
        pad=35,
    ):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += collection_time + learn_time
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(
            self.num_steps_per_env * self.env.num_envs / max(collection_time + learn_time, 1e-6)
        )
        if self.writer is not None:
            self.writer.add_scalar("Loss/value_function", mean_value_loss, it)
            self.writer.add_scalar("Loss/surrogate", mean_surrogate_loss, it)
            self.writer.add_scalar("Loss/bc_mse", mean_bc_loss, it)
            self.writer.add_scalar("Loss/bc_coef", bc_coef, it)
            self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), it)
            if len(rewbuffer) > 0:
                self.writer.add_scalar("Train/mean_reward", statistics.mean(rewbuffer), it)
                self.writer.add_scalar(
                    "Train/mean_episode_length", statistics.mean(lenbuffer), it
                )

        ep_string = ""
        if ep_infos:
            for key in ep_infos[0]:
                vals = []
                for ep_info in ep_infos:
                    v = ep_info[key]
                    if not isinstance(v, torch.Tensor):
                        v = torch.tensor([v], device=self.device)
                    vals.append(v.float().mean())
                value = torch.stack(vals).mean()
                if self.writer is not None:
                    self.writer.add_scalar(f"Episode/{key}", value, it)
                ep_string += f"{f'Mean episode {key}:':>{pad}} {value:.4f}\n"

        rew = statistics.mean(rewbuffer) if rewbuffer else float("nan")
        elen = statistics.mean(lenbuffer) if lenbuffer else float("nan")
        elapsed = max(it - start_iter + 1, 1)
        eta = self.tot_time / elapsed * max(tot_iter - it - 1, 0)
        title = f" Learning iteration {it}/{tot_iter} "
        print(
            f"{'#' * width}\n"
            f"{title.center(width, ' ')}\n\n"
            f"{'Computation:':>{pad}} {fps:.0f} steps/s "
            f"(collection: {collection_time:.3f}s, learning {learn_time:.3f}s)\n"
            f"{'Value function loss:':>{pad}} {mean_value_loss:.4f}\n"
            f"{'Surrogate loss:':>{pad}} {mean_surrogate_loss:.4f}\n"
            f"{'BC MSE loss:':>{pad}} {mean_bc_loss:.4f}\n"
            f"{'BC coef:':>{pad}} {bc_coef:.4f}\n"
            f"{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"
            f"{'Mean reward:':>{pad}} {rew:.2f}\n"
            f"{'Mean episode length:':>{pad}} {elen:.2f}\n"
            f"{ep_string}"
            f"{'-' * width}\n"
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
            f"{'Iteration time:':>{pad}} {collection_time + learn_time:.2f}s\n"
            f"{'Total time:':>{pad}} {self.tot_time:.2f}s\n"
            f"{'ETA:':>{pad}} {eta:.1f}s\n"
        )

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos or {"twist_student": True, "num_obs": self.STUDENT_OBS},
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in loaded:
            self.alg.optimizer.load_state_dict(loaded["optimizer_state_dict"])
        self.current_learning_iteration = int(loaded.get("iter", 0))
        return loaded.get("infos")
