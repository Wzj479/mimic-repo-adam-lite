#!/usr/bin/env python3
"""Sweep ankle PD gains for jump sim2sim; score by landing survival."""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CFG_DIR = Path(__file__).resolve().parent / "configs"
BASE_CFG = CFG_DIR / "adam_lite_jump.yaml"
DEPLOY_PY = Path(__file__).resolve().parent / "deploy_mujoco_jump.py"

# Joint order: L/R [hipPitch, hipRoll, hipYaw, kneePitch, anklePitch, ankleRoll]
IDX_AP = (4, 10)
IDX_AR = (5, 11)


def load_base():
    with open(BASE_CFG) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def apply_ankle(cfg, ap_kp, ap_kd, ar_kp, ar_kd):
    kps = list(cfg["kps"])
    kds = list(cfg["kds"])
    for i in IDX_AP:
        kps[i] = float(ap_kp)
        kds[i] = float(ap_kd)
    for i in IDX_AR:
        kps[i] = float(ar_kp)
        kds[i] = float(ar_kd)
    cfg["kps"] = kps
    cfg["kds"] = kds
    return cfg


def score_csv(path: Path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return dict(score=-1e9, collapse_t=0.0, zerr_pre=99.0, max_pitch=99.0, zerr_full=99.0)

    t = np.array([float(r["t"]) for r in rows])
    ze = np.array([float(r["z_err"]) for r in rows])
    pitch = np.array([float(r["pitch_deg"]) for r in rows])
    xy = np.hypot(
        np.array([float(r["root_x"]) - float(r["ref_x"]) for r in rows]),
        np.array([float(r["root_y"]) - float(r["ref_y"]) for r in rows]),
    )

    collapse_t = float(t[-1])
    for i in range(len(t)):
        if abs(ze[i]) > 0.25 and abs(pitch[i]) > 40:
            collapse_t = float(t[i])
            break
        if abs(pitch[i]) > 50:
            collapse_t = float(t[i])
            break

    pre = t < min(collapse_t, 6.0)
    if pre.sum() == 0:
        pre = t < 2.0
    land = (t >= 4.5) & (t < min(collapse_t + 0.2, 6.5))
    zerr_pre = float(np.mean(np.abs(ze[pre]))) if pre.any() else 99.0
    max_pitch = float(np.max(np.abs(pitch[land]))) if land.any() else float(np.max(np.abs(pitch)))
    xy_55 = float(xy[np.argmin(np.abs(t - 5.5))]) if len(t) else 99.0
    zerr_full = float(np.mean(np.abs(ze)))

    # Prefer longer survival, then lower pre-collapse z err, then lower landing pitch, then less XY drift.
    score = (
        collapse_t * 20.0
        - zerr_pre * 80.0
        - max_pitch * 0.15
        - min(xy_55, 1.5) * 10.0
        - zerr_full * 5.0
    )
    return dict(
        score=score,
        collapse_t=collapse_t,
        zerr_pre=zerr_pre,
        max_pitch=max_pitch,
        xy_55=xy_55,
        zerr_full=zerr_full,
    )


def run_one(cfg, tag: str, duration: float = 12.0):
    with tempfile.TemporaryDirectory(prefix="ankle_sweep_") as td:
        cfg_path = Path(td) / f"{tag}.yaml"
        log_csv = Path(td) / f"{tag}.csv"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        # Point deploy at temp config by basename under configs — copy into configs briefly.
        tmp_name = f"_sweep_{tag}.yaml"
        tmp_cfg = CFG_DIR / tmp_name
        tmp_cfg.write_text(cfg_path.read_text())
        env = os.environ.copy()
        env["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")
        env["LD_LIBRARY_PATH"] = (
            f"{env.get('CONDA_PREFIX','')}/lib:" + env.get("LD_LIBRARY_PATH", "")
        )
        cmd = [
            sys.executable,
            str(DEPLOY_PY),
            tmp_name,
            "--headless",
            f"--duration={duration}",
            f"--log-csv={log_csv}",
        ]
        try:
            subprocess.run(
                cmd,
                cwd=str(DEPLOY_PY.parent),
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            metrics = score_csv(log_csv)
        finally:
            if tmp_cfg.exists():
                tmp_cfg.unlink()
    return metrics


def main():
    base = load_base()
    # Ensure Jul14 policy path
    results = []

    # Round 1: coarse
    grid = []
    for ap_kd in [2.55, 4.0, 5.5, 7.0]:
        for ar_kp in [0.0, 6.0, 12.0]:
            for ap_kp in [25.0, 40.0]:
                for ar_kd in [0.35, 1.0]:
                    grid.append((ap_kp, ap_kd, ar_kp, ar_kd))

    print(f"[sweep] round1 candidates={len(grid)}", flush=True)
    for i, (ap_kp, ap_kd, ar_kp, ar_kd) in enumerate(grid):
        cfg = apply_ankle(deepcopy(base), ap_kp, ap_kd, ar_kp, ar_kd)
        tag = f"r1_{i:03d}_ap{ap_kp:g}_{ap_kd:g}_ar{ar_kp:g}_{ar_kd:g}".replace(".", "p")
        m = run_one(cfg, tag, duration=12.0)
        m.update(ap_kp=ap_kp, ap_kd=ap_kd, ar_kp=ar_kp, ar_kd=ar_kd, tag=tag)
        results.append(m)
        print(
            f"[r1 {i+1}/{len(grid)}] apKp={ap_kp:g} apKd={ap_kd:g} arKp={ar_kp:g} arKd={ar_kd:g} "
            f"-> collapse={m['collapse_t']:.2f}s zpre={m['zerr_pre']*100:.1f}cm "
            f"pitch={m['max_pitch']:.0f} score={m['score']:.1f}",
            flush=True,
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[:6]
    print("\n[sweep] round1 top:", flush=True)
    for m in top:
        print(
            f"  score={m['score']:.1f} collapse={m['collapse_t']:.2f} "
            f"ap={m['ap_kp']}/{m['ap_kd']} ar={m['ar_kp']}/{m['ar_kd']}",
            flush=True,
        )

    # Round 2: refine around best
    best0 = top[0]
    refine = []
    for dap_kp in [-10, 0, 10]:
        for dap_kd in [-1.0, 0, 1.0, 2.0]:
            for dar_kp in [-3, 0, 3, 6]:
                ap_kp = max(15.0, best0["ap_kp"] + dap_kp)
                ap_kd = max(1.5, best0["ap_kd"] + dap_kd)
                ar_kp = max(0.0, best0["ar_kp"] + dar_kp)
                ar_kd = best0["ar_kd"]
                key = (round(ap_kp, 2), round(ap_kd, 2), round(ar_kp, 2), round(ar_kd, 2))
                if key not in {(round(x["ap_kp"],2), round(x["ap_kd"],2), round(x["ar_kp"],2), round(x["ar_kd"],2)) for x in results}:
                    refine.append(key)

    print(f"\n[sweep] round2 refine={len(refine)}", flush=True)
    for i, (ap_kp, ap_kd, ar_kp, ar_kd) in enumerate(refine):
        cfg = apply_ankle(deepcopy(base), ap_kp, ap_kd, ar_kp, ar_kd)
        tag = f"r2_{i:03d}"
        m = run_one(cfg, tag, duration=14.0)
        m.update(ap_kp=ap_kp, ap_kd=ap_kd, ar_kp=ar_kp, ar_kd=ar_kd, tag=tag)
        results.append(m)
        print(
            f"[r2 {i+1}/{len(refine)}] apKp={ap_kp:g} apKd={ap_kd:g} arKp={ar_kp:g} arKd={ar_kd:g} "
            f"-> collapse={m['collapse_t']:.2f}s zpre={m['zerr_pre']*100:.1f}cm "
            f"pitch={m['max_pitch']:.0f} score={m['score']:.1f}",
            flush=True,
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    best = results[0]

    # Final verification with longer duration
    cfg = apply_ankle(deepcopy(base), best["ap_kp"], best["ap_kd"], best["ar_kp"], best["ar_kd"])
    m_final = run_one(cfg, "final_best", duration=20.0)
    print("\n========== BEST ==========", flush=True)
    print(
        f"anklePitch Kp/Kd = {best['ap_kp']}/{best['ap_kd']}\n"
        f"ankleRoll  Kp/Kd = {best['ar_kp']}/{best['ar_kd']}\n"
        f"collapse_t={m_final['collapse_t']:.2f}s "
        f"zerr_pre={m_final['zerr_pre']*100:.1f}cm "
        f"max_pitch={m_final['max_pitch']:.1f} "
        f"zerr_full={m_final['zerr_full']*100:.1f}cm "
        f"score={m_final['score']:.1f}",
        flush=True,
    )

    # Baseline compare
    cfg_b = deepcopy(base)
    m_base = run_one(cfg_b, "baseline", duration=20.0)
    print(
        f"BASELINE collapse_t={m_base['collapse_t']:.2f}s "
        f"zerr_pre={m_base['zerr_pre']*100:.1f}cm "
        f"max_pitch={m_base['max_pitch']:.1f} "
        f"zerr_full={m_base['zerr_full']*100:.1f}cm "
        f"score={m_base['score']:.1f}",
        flush=True,
    )

    out = {
        "best": {
            "ap_kp": best["ap_kp"],
            "ap_kd": best["ap_kd"],
            "ar_kp": best["ar_kp"],
            "ar_kd": best["ar_kd"],
            **{k: m_final[k] for k in ("collapse_t", "zerr_pre", "max_pitch", "zerr_full", "score")},
        },
        "baseline": {k: m_base[k] for k in ("collapse_t", "zerr_pre", "max_pitch", "zerr_full", "score")},
        "top5": [
            {k: r[k] for k in ("ap_kp", "ap_kd", "ar_kp", "ar_kd", "collapse_t", "zerr_pre", "max_pitch", "score")}
            for r in results[:5]
        ],
    }
    out_path = ROOT / "logs" / "adam_lite_jump" / "sim2sim" / "ankle_pd_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    out_path.write_text(json.dumps(out, indent=2))
    print(f"[sweep] wrote {out_path}", flush=True)

    # Write best into yaml if better than baseline
    if m_final["score"] > m_base["score"] + 0.5:
        apply_ankle(base, best["ap_kp"], best["ap_kd"], best["ar_kp"], best["ar_kd"])
        # Preserve list formatting roughly via dump of only kps/kds update
        with open(BASE_CFG) as f:
            text = f.read()
        # Replace kps/kds lines
        import re

        kps = base["kps"]
        kds = base["kds"]
        kps_s = "kps: [" + ", ".join(str(int(x)) if float(x).is_integer() else str(x) for x in kps) + "]"
        kds_s = "kds: [" + ", ".join(str(x) for x in kds) + "]"
        text2 = re.sub(r"^kps: \[.*?\]$", kps_s, text, flags=re.M)
        text2 = re.sub(r"^kds: \[.*?\]$", kds_s, text2, flags=re.M)
        # comment
        if "Ankle PD tuned" not in text2:
            text2 = text2.replace(
                "# Matched to Jul14 teacher_ft policy (ankleRoll Kp=0). Stab finetune uses adam_lite_jump_stab.yaml.",
                "# Ankle PD tuned via sweep_ankle_pd.py for landing pitch survival (Jul14 policy).",
            )
        BASE_CFG.write_text(text2)
        print(f"[sweep] UPDATED {BASE_CFG}", flush=True)
    else:
        print("[sweep] no improvement over baseline; yaml unchanged", flush=True)


if __name__ == "__main__":
    main()
