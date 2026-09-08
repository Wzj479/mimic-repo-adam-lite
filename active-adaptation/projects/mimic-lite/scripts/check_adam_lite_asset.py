#!/usr/bin/env python3
"""CPU smoke check for the native Adam Lite 12DOF MimicLite asset. No GPU required."""

from __future__ import annotations

from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "mimic_lite/assets/adam_lite/adam_lite_12dof.xml"

EXPECTED_ACTUATED = [
    "hipPitch_Left",
    "hipRoll_Left",
    "hipYaw_Left",
    "kneePitch_Left",
    "anklePitch_Left",
    "ankleRoll_Left",
    "hipPitch_Right",
    "hipRoll_Right",
    "hipYaw_Right",
    "kneePitch_Right",
    "anklePitch_Right",
    "ankleRoll_Right",
]


def check_mjcf() -> None:
    print(f"xml={XML}")
    if not XML.is_file():
        raise SystemExit(f"missing MJCF: {XML}")
    model = mujoco.MjModel.from_xml_path(str(XML))
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)
    ]
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)
    ]
    geom_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)
    ]
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody} njnt={model.njnt}")
    print("joints:", joint_names)
    print("bodies:", body_names)
    print(
        "collision:",
        [name for name in geom_names if name and name.endswith("_collision")],
    )
    missing = [name for name in EXPECTED_ACTUATED if name not in joint_names]
    if missing:
        raise SystemExit(f"missing actuated joints: {missing}")
    if "pelvis" not in body_names or "toeLeft" not in body_names or "toeRight" not in body_names:
        raise SystemExit("missing expected bodies")
    if model.nu != 0:
        raise SystemExit(f"expected stripped XML actuators, got nu={model.nu}")
    print("MJCF load OK")


def check_registry() -> None:
    from active_adaptation.registry import Registry
    from mimic_lite.assets import adam_lite

    registry = Registry.instance()
    factory = registry.get("asset", "adam_lite_12dof")
    print("registered", factory, "name=adam_lite_12dof")
    print("mjcf_path", adam_lite.ADAM_LITE_MJCF_PATH)
    print("joint_names_simulation", adam_lite.ADAM_LITE_JOINT_NAMES)
    if list(adam_lite.ADAM_LITE_JOINT_NAMES) != EXPECTED_ACTUATED:
        raise SystemExit("joint_names_simulation mismatch")
    init = adam_lite.ADAM_LITE_INIT_STATE.joint_pos
    expected_pose = {
        "hipPitch_Left": -0.32,
        "kneePitch_Left": 0.66,
        "hipYaw_Right": 0.18,
    }
    for name, value in expected_pose.items():
        if abs(float(init[name]) - value) > 1e-6:
            raise SystemExit(f"init pose mismatch {name}: {init[name]} != {value}")
    print("registry OK")


def check_mjlab_spec() -> None:
    from active_adaptation.assets.asset_cfg import coerce_asset_spec
    from active_adaptation.registry import Registry

    spec = coerce_asset_spec(Registry.instance().get("asset", "adam_lite_12dof"), backend="mjlab")
    actuators = spec.config.articulation.actuators
    print("mjlab spec", type(spec.config).__name__, "sensors", len(spec.sensors), "actuator_groups", len(actuators))
    if len(actuators) != 6:
        raise SystemExit(f"expected 6 actuator groups, got {len(actuators)}")
    model = spec.config.spec_fn().compile()
    print(
        "compiled nq",
        model.nq,
        "nv",
        model.nv,
        "nu",
        model.nu,
        "njnt",
        model.njnt,
    )
    print("actuator targets:", [a.target_names_expr for a in actuators])
    print("mjlab spec OK")


def main() -> None:
    check_mjcf()
    import active_adaptation as aa

    if not aa.get_backend():
        aa.set_backend("mjlab")
    check_registry()
    check_mjlab_spec()
    print("stage 1 asset checks passed")


if __name__ == "__main__":
    main()
