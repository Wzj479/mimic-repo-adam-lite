# PND RL Gym 核心流程分析

> 快速理解基于强化学习的腿式机器人运动控制系统

---

## 📋 目录

1. [系统概述](#系统概述)
2. [整体流程](#整体流程)
3. [五大核心步骤](#五大核心步骤)
4. [关键技术亮点](#关键技术亮点)
5. [代码结构](#代码结构)
6. [快速上手](#快速上手)

---

## 🎯 系统概述

这是一个基于**强化学习（Reinforcement Learning）**的腿式机器人运动控制系统：

- **仿真器**：Isaac Gym (NVIDIA) - 支持GPU加速的并行仿真
- **算法**：PPO (Proximal Policy Optimization) - 稳定高效的策略梯度算法
- **特点**：支持数千个环境并行训练，大幅缩短训练时间
- **目标**：让机器人通过试错学会稳定行走、转向等运动技能

**核心理念**：通过在仿真器中让机器人"大量试错"，用奖励函数引导它学会正确的行走方式。

---

## 🔄 整体流程

```
训练(Train) → 验证(Play) → 跨仿真器部署(Sim2Sim) → 实体机器人部署(Sim2Real)
     ↓              ↓                ↓                      ↓
  Isaac Gym      Isaac Gym         Mujoco            真实机器人硬件
  并行训练        可视化测试        验证迁移性           最终部署
```

### 各阶段说明

| 阶段 | 命令 | 目的 | 关键点 |
|------|------|------|--------|
| **Train** | `python legged_gym/scripts/train.py --task=adam_lite_12dof` | 训练策略网络 | 4096+个并行环境，加速收敛 |
| **Play** | `python legged_gym/scripts/play.py --task=adam_lite_12dof` | 可视化验证 | 检查训练效果，导出模型 |
| **Sim2Sim** | `python deploy/deploy_mujoco/deploy_mujoco.py` | Mujoco测试 | 验证跨仿真器泛化能力 |
| **Sim2Real** | `python deploy/deploy_real/deploy_real.py` | 真机部署 | 最终的现实世界验证 |

---

## 🔑 五大核心步骤

### 1️⃣ 环境配置 (Configuration)

**核心文件**：`legged_gym/envs/adam_lite_12dof/adam_lite_12dof_config.py`

这是整个系统的"大脑配置"，定义训练的所有关键参数。

#### 主要配置内容

**a) 初始状态配置**
```python
class init_state:
    pos = [0.0, 0.0, 0.85]  # 初始位置 (x, y, z) 米
    default_joint_angles = {  # 默认关节角度
        'hipPitch_Left': -0.1,
        'kneePitch_Left': 0.3,
        'anklePitch_Left': -0.2,
        # ... 其他关节
    }
```

**b) PD控制器参数**
```python
class control:
    control_type = 'P'
    stiffness = {'hipPitch': 150, 'kneePitch': 180, ...}  # 刚度
    damping = {'hipPitch': 3.0, 'kneePitch': 4.0, ...}    # 阻尼
    action_scale = 0.22  # 动作缩放
    decimation = 4       # 控制频率分频
```

**关键公式**：
```
目标关节角度 = action_scale × action + default_joint_angle
关节力矩 = Kp × (目标角度 - 当前角度) + Kd × (0 - 当前角速度)
```

**c) 奖励函数权重**
```python
class rewards.scales:
    tracking_lin_vel = 1.0        # 跟踪线速度指令（重要！）
    tracking_ang_vel = 0.5        # 跟踪角速度指令
    orientation = -0.5            # 保持正确姿态
    base_height = -3.0            # 保持目标高度
    collision = -1.0              # 惩罚碰撞
    feet_swing_height = -10.0     # 脚抬起高度控制
    feet_lateral_deviation = -5.0 # 防止外八字
    alive = 0.15                  # 存活奖励
```

**d) PPO算法参数**
```python
class algorithm:
    learning_rate = 8e-4          # 学习率
    num_learning_epochs = 3       # 每次更新的训练轮数
    num_mini_batches = 8          # 小批量数量
    clip_param = 0.15             # PPO裁剪参数

class policy:
    actor_hidden_dims = [128, 64] # Actor网络结构
    critic_hidden_dims = [128, 64]# Critic网络结构
    rnn_type = 'lstm'             # 使用LSTM记忆
    rnn_hidden_size = 128         # LSTM隐藏层大小
```

---

### 2️⃣ 环境交互循环 (Environment Step)

**核心文件**：`legged_gym/envs/base/legged_robot.py`

每一步仿真的完整流程：

```python
def step(actions):
    """
    输入：actions [num_envs, num_actions] - 来自策略网络的动作
    输出：observations, rewards, dones, infos
    """
    
    # 步骤1: 裁剪动作到合理范围
    actions = clip(actions, -clip_actions, clip_actions)
    
    # 步骤2: 执行decimation次物理仿真（提高控制频率）
    for _ in range(decimation):  # decimation = 4
        # 2.1 计算关节力矩（PD控制）
        torques = compute_torques(actions)
        
        # 2.2 应用力矩到仿真器
        gym.set_dof_actuation_force_tensor(torques)
        
        # 2.3 推进物理仿真一步
        gym.simulate()
        
        # 2.4 刷新关节状态
        gym.refresh_dof_state_tensor()
    
    # 步骤3: 后处理（更新状态、计算奖励）
    post_physics_step()
        ├─ 更新机器人状态（位置、速度、姿态）
        ├─ 更新脚部状态
        ├─ 计算观测值 → compute_observations()
        ├─ 计算奖励 → compute_reward()
        └─ 检查重置条件 → check_termination()
    
    # 步骤4: 返回结果
    return observations, rewards, dones, infos
```

**关键概念**：
- **Decimation**：策略网络以50Hz运行，物理仿真以200Hz运行（4倍细化）
- **并行化**：同时运行4096个环境，每个环境独立计算
- **GPU加速**：所有张量运算在GPU上完成，极大提升效率

---

### 3️⃣ 观测值构建 (Observation)

**核心文件**：`legged_gym/envs/adam_lite_12dof/adam_lite_12dof_env.py`

**机器人"看到"什么信息？**

```python
def compute_observations(self):
    """
    构建47维观测向量
    """
    # 步态相位（用于引导周期性运动）
    sin_phase = sin(2π × phase)
    cos_phase = cos(2π × phase)
    
    # 组装观测向量
    obs = [
        base_ang_vel,              # 3维：基座角速度 [ωx, ωy, ωz]
        projected_gravity,         # 3维：重力方向 [gx, gy, gz]
        commands,                  # 3维：速度指令 [vx_cmd, vy_cmd, ωz_cmd]
        dof_pos - default_pos,     # 12维：关节角度偏差
        dof_vel,                   # 12维：关节速度
        previous_actions,          # 12维：上一步动作
        sin_phase, cos_phase       # 2维：步态相位
    ]  # 总计：3+3+3+12+12+12+2 = 47维
    
    # 添加噪声（提高鲁棒性）
    if add_noise:
        obs += uniform_noise(-noise_level, noise_level)
    
    return obs
```

**为什么需要这些信息？**

| 观测项 | 用途 | 重要性 |
|--------|------|--------|
| 角速度 | 感知旋转状态，保持平衡 | ⭐⭐⭐⭐⭐ |
| 重力方向 | 判断身体倾斜，控制姿态 | ⭐⭐⭐⭐⭐ |
| 速度指令 | 知道目标速度，跟踪指令 | ⭐⭐⭐⭐⭐ |
| 关节位置 | 了解当前姿态 | ⭐⭐⭐⭐ |
| 关节速度 | 预测下一步状态 | ⭐⭐⭐⭐ |
| 上一步动作 | 保持动作连续性 | ⭐⭐⭐ |
| 步态相位 | 引导左右腿交替摆动 | ⭐⭐⭐⭐ |

**步态相位详解**：
```
period = 0.8秒（完整步态周期）
phase_left = (current_time % period) / period      # 左腿相位 [0, 1)
phase_right = (phase_left + 0.5) % 1               # 右腿相位（相差半个周期）

当 phase < 0.55 → 支撑相（脚接触地面）
当 phase ≥ 0.55 → 摆动相（脚在空中）
```

---

### 4️⃣ 奖励计算 (Reward Shaping)

**核心文件**：`legged_gym/envs/adam_lite_12dof/adam_lite_12dof_env.py`

**这是强化学习的核心——告诉AI什么是"好"的行为！**

#### 主要奖励项

**a) 速度跟踪奖励（最重要）**
```python
def _reward_tracking_lin_vel(self):
    """奖励：跟踪线速度指令"""
    lin_vel_error = torch.sum(torch.square(
        self.commands[:, :2] - self.base_lin_vel[:, :2]
    ), dim=1)
    return torch.exp(-lin_vel_error / 0.25)
```
权重：`1.0` （最高优先级）

**b) 步态协调奖励**
```python
def _reward_contact(self):
    """奖励：脚在正确时机接触地面"""
    res = 0
    for foot in feet:
        is_stance = leg_phase[foot] < 0.55      # 应该接触
        has_contact = contact_force[foot] > 1    # 实际接触
        res += (is_stance == has_contact)        # 一致则奖励
    return res
```
权重：`0.2`

**c) 脚部摆动控制**
```python
def _reward_feet_swing_height(self):
    """惩罚：摆动腿高度不足"""
    target_height = 0.1  # 目标离地10cm
    error = (feet_height - target_height)² × is_swing
    return sum(error)
```
权重：`-10.0` （负值表示惩罚）

**d) 姿态控制**
```python
def _reward_orientation(self):
    """惩罚：身体倾斜"""
    # projected_gravity在正确姿态下应为[0, 0, -1]
    return sum(square(projected_gravity[:, :2]))
```
权重：`-0.5`

**e) 防止外八字**
```python
def _reward_feet_lateral_deviation(self):
    """惩罚：脚的朝向偏离前进方向"""
    yaw_diff = foot_yaw - body_yaw  # 脚与身体的偏航角差
    return sum(square(yaw_diff))
```
权重：`-5.0`

**f) 碰撞惩罚**
```python
def _reward_collision(self):
    """惩罚：膝盖、骨盆触地"""
    return sum(contact_forces[unwanted_parts])
```
权重：`-1.0`

**g) 存活奖励**
```python
def _reward_alive(self):
    """奖励：每步存活奖励"""
    return 1.0
```
权重：`0.15`

#### 总奖励计算

```python
total_reward = sum(scale[i] × reward[i] for all rewards)

# 例如：
reward = 1.0 × tracking_vel 
       + 0.5 × tracking_ang 
       - 0.5 × orientation_error
       - 3.0 × height_error
       + 0.2 × correct_contact
       - 10.0 × swing_height_error
       - 5.0 × lateral_deviation
       - 1.0 × collision
       + 0.15 × alive
       ...
```

**奖励设计原则**：
1. ✅ **正奖励**：引导目标行为（速度跟踪、步态协调）
2. ❌ **负奖励**：抑制不良行为（碰撞、能耗、姿态不稳）
3. ⚖️ **权重平衡**：避免某个奖励项主导，导致忽视其他行为

---

### 5️⃣ PPO训练循环 (Training Loop)

**核心文件**：`legged_gym/scripts/train.py` 和 `rsl_rl` 库

#### 完整训练流程

```python
def train():
    # ========== 初始化 ==========
    # 1. 创建并行环境
    env = make_env(task='adam_lite_12dof')  # 4096个并行环境
    
    # 2. 创建PPO Runner
    ppo_runner = make_alg_runner(env)
        ├─ Actor网络 (策略网络)：obs → actions
        ├─ Critic网络 (价值网络)：obs → value
        └─ 优化器：Adam optimizer
    
    # 3. 初始化WandB日志（可选）
    wandb_logger = create_wandb_logger()
    
    # ========== 训练循环 ==========
    for iteration in range(max_iterations):  # max_iterations = 10000
        
        # ===== 阶段A: 数据收集 =====
        for step in range(num_steps_per_env):  # 32步
            # A1. 策略网络推理
            with torch.no_grad():
                actions, values = policy.act(observations)
            
            # A2. 环境交互
            next_obs, rewards, dones, infos = env.step(actions)
            
            # A3. 存储经验
            rollout_buffer.add(
                obs=observations,
                actions=actions,
                rewards=rewards,
                values=values,
                dones=dones
            )
            
            # A4. 自动重置完成的环境
            if any(dones):
                env.reset_idx(done_indices)
            
            observations = next_obs
        
        # ===== 阶段B: 策略更新 =====
        # B1. 计算优势函数（Advantage）
        advantages = compute_gae(
            rewards, values, dones,
            gamma=0.99,      # 折扣因子
            lambda=0.95      # GAE参数
        )
        
        # B2. 多轮更新
        for epoch in range(num_learning_epochs):  # 3轮
            # B2.1 打乱数据
            indices = shuffle(rollout_buffer)
            
            # B2.2 小批量更新
            for batch in split_batches(indices, num_mini_batches=8):
                # 重新计算当前策略下的值
                new_actions, new_values, log_probs = policy.evaluate(
                    rollout_buffer.obs[batch]
                )
                
                # 计算损失
                ratio = exp(log_probs - old_log_probs[batch])
                surr1 = ratio × advantages[batch]
                surr2 = clip(ratio, 1-ε, 1+ε) × advantages[batch]
                
                actor_loss = -min(surr1, surr2).mean()
                critic_loss = mse(new_values, returns[batch])
                entropy_loss = -entropy.mean()
                
                total_loss = actor_loss + 0.5×critic_loss + 0.02×entropy_loss
                
                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                clip_grad_norm_(parameters, max_grad_norm=1.0)
                optimizer.step()
        
        # ===== 阶段C: 记录与保存 =====
        if iteration % save_interval == 0:
            save_model(f'model_{iteration}.pt')
            log_metrics(wandb_logger)
    
    # ========== 训练结束 ==========
    export_policy('policy_lstm_1.pt')
```

#### 关键概念解释

**PPO算法核心思想**：
```
目标：最大化 E[min(ratio × A, clip(ratio, 1-ε, 1+ε) × A)]

其中：
- ratio = π_new(a|s) / π_old(a|s)  # 新旧策略的比率
- A = 优势函数（衡量动作比平均好多少）
- ε = 0.15  # 裁剪范围，防止策略更新过大
- clip确保策略不会变化太快，保持训练稳定
```

**优势函数（GAE）**：
```python
# Generalized Advantage Estimation
δ_t = r_t + γ×V(s_{t+1}) - V(s_t)  # 时序差分误差
A_t = δ_t + γλ×δ_{t+1} + (γλ)²×δ_{t+2} + ...

# 作用：告诉网络某个动作相比平均水平好多少
```

**并行化效率**：
```
单环境训练：1个机器人 × 32步 × 10000轮 = 320,000 步
并行训练：4096个机器人 × 32步 × 10000轮 = 131,072,000 步

加速比：~400倍！（实际考虑通信开销约100-200倍）
```

---

## 🚀 关键技术亮点

### 1. 大规模并行仿真

- **Isaac Gym**：NVIDIA开发的GPU加速仿真器
- **并行数量**：4096个环境同时运行
- **性能**：单卡RTX 3090可达 100,000+ FPS（所有环境总和）
- **优势**：大幅缩短训练时间（数小时而非数天）

```python
# 传统方式：串行
for env in envs:
    obs, reward = env.step(action)  # CPU，慢

# Isaac Gym：并行
obs, rewards = envs.step(actions)  # GPU张量操作，快
```

---

### 2. 步态相位引导

**问题**：如何让机器人学会"左右脚交替"行走？

**方案**：在观测中加入步态相位信息

```python
period = 0.8秒
phase_left = (time % period) / period        # 左腿：0 → 1
phase_right = (phase_left + 0.5) % 1         # 右腿：0.5 → 1 → 0 → 0.5

obs += [sin(2π × phase), cos(2π × phase)]   # 编码为连续信号
```

**效果**：
- ✅ 提供周期性信号，引导周期性运动
- ✅ 左右腿相位差0.5，自然产生交替摆动
- ✅ sin/cos编码保证相位连续性（0°和360°接近）

---

### 3. 域随机化 (Domain Randomization)

**问题**：仿真器训练的策略在真实世界失效（Sim2Real Gap）

**方案**：训练时随机化物理参数

```python
class domain_rand:
    randomize_friction = True
    friction_range = [0.1, 1.25]      # 摩擦系数：光滑冰面 → 粗糙地面
    
    randomize_base_mass = True
    added_mass_range = [-0.5, 0.5]    # 质量：±0.5kg
    
    push_robots = True
    max_push_vel_xy = 1.5             # 随机推力：模拟外部扰动
```

**效果**：策略在各种条件下都训练过，真机部署时更鲁棒

---

### 4. PD控制器 (Position Control)

**为什么不直接输出力矩？**

| 方法 | 优点 | 缺点 |
|------|------|------|
| 直接力矩控制 | 精细控制 | 难以稳定，需要精确动力学模型 |
| **PD位置控制** | **稳定，易于迁移** | **响应稍慢** |

**PD控制公式**：
```python
target_pos = action_scale × action + default_joint_angle
torque = Kp × (target_pos - current_pos) + Kd × (0 - current_vel)
```

**类比**：
- **Kp（比例项）**：弹簧刚度，角度差越大力越大
- **Kd（微分项）**：阻尼器，速度越大阻力越大
- 结果：关节会"弹性"地到达目标角度

---

### 5. Curriculum Learning（课程学习）

**问题**：初期随机策略无法完成复杂任务（如高速行走）

**方案**：从简单任务开始，逐步增加难度

```python
class commands:
    curriculum = True
    # 速度指令范围随训练进度扩大
    initial_range: lin_vel_x = [0.0, 0.3]  # 初期：慢速
    final_range:   lin_vel_x = [0.0, 0.8]  # 后期：快速
```

**进度调整**：
```python
# 根据成功率自动调整难度
if success_rate > 0.8:
    increase_difficulty()
elif success_rate < 0.5:
    decrease_difficulty()
```

---

### 6. LSTM记忆网络

**问题**：单步观测无法判断运动趋势（如"正在加速"还是"正在减速"）

**方案**：使用LSTM保留历史信息

```python
class policy:
    policy_class_name = "ActorCriticRecurrent"  # 而非普通MLP
    rnn_type = 'lstm'
    rnn_hidden_size = 128
```

**网络结构**：
```
obs[t] ──┐
         ├─→ [LSTM] ──→ hidden_state[t] ──→ [MLP] ──→ action[t]
hidden[t-1] ──┘
```

**优势**：
- ✅ 记住过去几步的状态变化趋势
- ✅ 更平滑的动作输出
- ✅ 更好的速度跟踪性能

---

## 📁 代码结构

```
pnd_rl_gym/
├── legged_gym/                        # 核心训练代码
│   ├── envs/                          # 环境定义
│   │   ├── base/
│   │   │   ├── legged_robot.py        # 【基础环境】step, reset, compute_reward
│   │   │   ├── legged_robot_config.py # 基础配置类
│   │   │   └── base_task.py           # 任务基类
│   │   └── adam_lite_12dof/           # Adam Lite 12DOF机器人
│   │       ├── adam_lite_12dof_env.py       # 【环境实现】观测、奖励函数
│   │       ├── adam_lite_12dof_config.py    # 【配置文件】所有超参数
│   │       └── __init__.py
│   ├── scripts/
│   │   ├── train.py                   # 【训练入口】
│   │   └── play.py                    # 【测试入口】可视化+导出模型
│   └── utils/                         # 工具函数
│       ├── task_registry.py           # 任务注册器
│       ├── helpers.py                 # 辅助函数
│       └── wandb_runner.py            # WandB集成
│
├── rsl_rl/                            # PPO算法实现（外部库）
│   ├── algorithms/
│   │   └── ppo.py                     # PPO算法核心
│   ├── modules/
│   │   ├── actor_critic.py            # Actor-Critic网络
│   │   └── actor_critic_recurrent.py  # LSTM版本
│   └── runners/
│       └── on_policy_runner.py        # 训练循环
│
├── resources/                         # 机器人资源
│   └── robots/
│       └── adam_lite_12dof/
│           ├── adam_lite_12dof.urdf   # 机器人模型
│           └── meshes/                # 网格文件
│
├── deploy/                            # 部署代码
│   ├── deploy_mujoco/                 # 【Sim2Sim】Mujoco部署
│   │   ├── deploy_mujoco.py
│   │   └── configs/
│   │       └── adam_lite_12dof.yaml
│   └── deploy_real/                   # 【Sim2Real】真机部署
│       ├── deploy_real.py
│       ├── cpp_g1/                    # C++部署示例
│       └── configs/
│
└── logs/                              # 训练日志
    └── adam_lite_12dof/
        └── {timestamp}/
            ├── model_*.pt             # 训练检查点
            ├── config.json            # 配置快照
            └── code_snapshot/         # 代码快照
```

### 关键文件说明

| 文件 | 作用 | 核心内容 |
|------|------|----------|
| `adam_lite_12dof_config.py` | 配置中心 | 奖励权重、PD参数、PPO超参数 |
| `adam_lite_12dof_env.py` | 环境逻辑 | 观测构建、奖励函数实现 |
| `legged_robot.py` | 环境基类 | step循环、物理仿真接口 |
| `train.py` | 训练脚本 | 环境创建、PPO训练循环 |
| `play.py` | 测试脚本 | 模型加载、可视化、导出 |

---

## 🎓 快速上手

### 环境安装

```bash
# 1. 创建conda环境
conda create -n pndbotics_rl python=3.8
conda activate pndbotics_rl

# 2. 安装PyTorch（根据CUDA版本）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. 安装Isaac Gym（需要下载）
cd isaacgym/python
pip install -e .

# 4. 安装本项目
cd pnd_rl_gym
pip install -e .
```

### 训练命令

```bash
# 基础训练（无界面，最快）
python legged_gym/scripts/train.py --task=adam_lite_12dof --headless

# 带可视化训练（实时查看训练过程，慢）
python legged_gym/scripts/train.py --task=adam_lite_12dof

# 启用WandB日志
python legged_gym/scripts/train.py --task=adam_lite_12dof --headless --wandb

# 从检查点继续训练
python legged_gym/scripts/train.py --task=adam_lite_12dof --resume --checkpoint=500
```

### 测试命令

```bash
# 可视化最新模型
python legged_gym/scripts/play.py --task=adam_lite_12dof

# 指定检查点
python legged_gym/scripts/play.py --task=adam_lite_12dof --checkpoint=800

# 测试默认姿态（所有动作为0）
python legged_gym/scripts/play.py --task=adam_lite_12dof --test_default_pose
```

### 部署命令

```bash
# Mujoco部署
python deploy/deploy_mujoco/deploy_mujoco.py adam_lite_12dof.yaml

# 真机部署（需要实体机器人）
python deploy/deploy_real/deploy_real.py eth0 adam_lite_12dof.yaml
```

---

## 📊 训练监控

### 关键指标

| 指标 | 说明 | 目标值 |
|------|------|--------|
| `mean_reward` | 平均回合奖励 | 持续上升 |
| `mean_episode_length` | 平均存活步数 | 接近最大值 |
| `tracking_lin_vel` | 速度跟踪误差 | 接近0 |
| `collision` | 碰撞次数 | 接近0 |
| `policy_loss` | 策略损失 | 稳定下降 |

### 训练时间估计

| 硬件配置 | 训练时间（10000轮） |
|----------|---------------------|
| RTX 4090 | ~3-4小时 |
| RTX 3090 | ~5-6小时 |
| RTX 3080 | ~7-8小时 |
| RTX 3070 | ~10-12小时 |

---

## 🔧 常见调试技巧

### 1. 机器人站不起来
- 检查 `default_joint_angles` 是否合理
- 增大 `alive` 奖励权重
- 减小 `base_height` 惩罚权重

### 2. 速度跟踪不准
- 增大 `tracking_lin_vel` 权重
- 检查 `commands_scale` 是否正确
- 减小 `action_rate` 惩罚（允许更大动作变化）

### 3. 步态不协调
- 增大 `contact` 奖励权重
- 调整 `phase_offset`（左右脚相位差）
- 增大 `feet_swing_height` 惩罚

### 4. 训练不稳定
- 减小 `learning_rate`
- 增大 `num_mini_batches`
- 减小 `clip_param`

---

## 📚 进阶学习资源

### 论文
- **PPO算法**：[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **腿式机器人RL**：[Learning Quadrupedal Locomotion over Challenging Terrain](https://arxiv.org/abs/2010.11251)
- **Domain Randomization**：[Sim-to-Real: Learning Agile Locomotion For Quadruped Robots](https://arxiv.org/abs/1804.10332)

### 相关项目
- [legged_gym](https://github.com/leggedrobotics/legged_gym) - 本项目基础
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl) - PPO实现
- [Isaac Gym](https://developer.nvidia.com/isaac-gym) - 仿真器文档

---

## ❓ FAQ

**Q: 为什么用PPO而不是SAC/TD3？**
A: PPO更稳定，适合高维动作空间；SAC适合连续控制但样本效率较低。

**Q: 能在CPU上训练吗？**
A: 可以但极慢（100倍+），强烈建议使用GPU。

**Q: 训练好的模型能直接用于其他机器人吗？**
A: 不能，需要重新训练。但可以复用奖励函数和超参数配置。

**Q: 如何加速训练？**
A: 增加并行环境数（`num_envs`）、关闭可视化（`--headless`）、使用更强GPU。

**Q: Sim2Real效果不好怎么办？**
A: 增强Domain Randomization、在真机上微调（Fine-tuning）、使用更精确的系统辨识。

---

## 🎉 总结

这套代码的核心思想：

1. **大规模并行仿真** → 加速数据收集
2. **精心设计的奖励函数** → 引导正确行为
3. **稳定的PPO算法** → 高效学习策略
4. **Domain Randomization** → 提升真机泛化
5. **PD控制 + 步态引导** → 简化学习难度

通过这些技术的组合，机器人可以在几小时内学会稳定行走！

---

**文档生成时间**：2025-10-13  
**适用版本**：pndbotics rl Gym v1.0  
**维护者**：AI Assistant

如有问题，请参考：
- 📖 [README.md](README.md)
- 🇨🇳 [README_zh.md](README_zh.md)
- 🔗 [GitHub Issues](https://github.com/pndbotics/pnd_rl_gym/issues)

