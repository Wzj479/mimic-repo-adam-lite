"""Native 12DOF PND Adam Lite asset for MimicLite.

This is the Adam Lite robot, not G1. G1/Atom files are only used as the
MimicLite registration pattern (AssetCfg + registry).
"""

from pathlib import Path

import active_adaptation.utils.symmetry as symmetry_utils
from active_adaptation.assets.asset_cfg import (
    ActuatorCfg,
    AssetCfg,
    InitialStateCfg,
)
from active_adaptation.registry import Registry

registry = Registry.instance()

_ASSET_DIR = Path(__file__).resolve().parent
ADAM_LITE_MJCF_PATH = _ASSET_DIR / "adam_lite_12dof.xml"
ADAM_LITE_URDF_PATH = _ASSET_DIR / "adam_lite_12dof.urdf"

# Actuated joints only. Waist and arms stay equality-fixed in the MJCF.
ADAM_LITE_JOINT_NAMES = [
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

ADAM_LITE_BODY_NAMES = [
    "pelvis",
    "hipPitchLeft",
    "hipRollLeft",
    "thighLeft",
    "shinLeft",
    "anklePitchLeft",
    "toeLeft",
    "hipPitchRight",
    "hipRollRight",
    "thighRight",
    "shinRight",
    "anklePitchRight",
    "toeRight",
    "waistRoll",
    "waistPitch",
    "torso",
    "shoulderPitchLeft",
    "shoulderRollLeft",
    "shoulderYawLeft",
    "elbowLeft",
    "wristYawLeft",
    "shoulderPitchRight",
    "shoulderRollRight",
    "shoulderYawRight",
    "elbowRight",
    "wristYawRight",
]

# Standing pose from pnd_jump Adam Lite jump training.
ADAM_LITE_INIT_STATE = InitialStateCfg(
    pos=(0.0, 0.0, 0.92),
    joint_pos={
        "hipPitch_Left": -0.32,
        "hipRoll_Left": 0.0,
        "hipYaw_Left": -0.18,
        "kneePitch_Left": 0.66,
        "anklePitch_Left": -0.29,
        "ankleRoll_Left": 0.0,
        "hipPitch_Right": -0.32,
        "hipRoll_Right": 0.0,
        "hipYaw_Right": 0.18,
        "kneePitch_Right": 0.66,
        "anklePitch_Right": -0.29,
        "ankleRoll_Right": 0.0,
        "elbow_Left": -1.6,
        "elbow_Right": -1.6,
        ".*": 0.0,
    },
    joint_vel={".*": 0.0},
)


def _motor_actuator(
    joint_names_expr: str,
    *,
    effort: float,
    velocity: float,
    stiffness: float,
    damping: float,
    friction: float = 0.01,
    armature: float = 0.01,
) -> ActuatorCfg:
    return ActuatorCfg(
        joint_names_expr=joint_names_expr,
        effort_limit=effort,
        velocity_limit=velocity,
        stiffness=stiffness,
        damping=damping,
        friction=friction,
        armature=armature,
    )


ADAM_LITE_ACTUATORS = {
    "hip_pitch": _motor_actuator(
        "hipPitch_Left|hipPitch_Right",
        effort=230.0,
        velocity=15.0,
        stiffness=305.0,
        damping=6.1,
    ),
    "hip_roll": _motor_actuator(
        "hipRoll_Left|hipRoll_Right",
        effort=160.0,
        velocity=8.0,
        stiffness=700.0,
        damping=30.0,
    ),
    "hip_yaw": _motor_actuator(
        "hipYaw_Left|hipYaw_Right",
        effort=105.0,
        velocity=8.0,
        stiffness=405.0,
        damping=6.1,
    ),
    "knee": _motor_actuator(
        "kneePitch_Left|kneePitch_Right",
        effort=230.0,
        velocity=15.0,
        stiffness=305.0,
        damping=6.1,
    ),
    "ankle_pitch": _motor_actuator(
        "anklePitch_Left|anklePitch_Right",
        effort=40.0,
        velocity=20.0,
        stiffness=25.0,
        damping=3.0,
    ),
    "ankle_roll": _motor_actuator(
        "ankleRoll_Left|ankleRoll_Right",
        effort=12.0,
        velocity=20.0,
        # pnd_jump uses Kp=0; mjlab position actuators divide effort by
        # stiffness, so 0 is illegal. Keep a weak gain instead.
        stiffness=1.0,
        damping=0.35,
    ),
}

JOINT_SYMMETRY_MAPPING = symmetry_utils.mirrored(
    {
        "hipPitch_Left": (1, "hipPitch_Right"),
        "hipRoll_Left": (-1, "hipRoll_Right"),
        "hipYaw_Left": (-1, "hipYaw_Right"),
        "kneePitch_Left": (1, "kneePitch_Right"),
        "anklePitch_Left": (1, "anklePitch_Right"),
        "ankleRoll_Left": (-1, "ankleRoll_Right"),
    }
)

SPATIAL_SYMMETRY_MAPPING = symmetry_utils.mirrored(
    {
        "pelvis": "pelvis",
        "hipPitchLeft": "hipPitchRight",
        "hipRollLeft": "hipRollRight",
        "thighLeft": "thighRight",
        "shinLeft": "shinRight",
        "anklePitchLeft": "anklePitchRight",
        "toeLeft": "toeRight",
        "waistRoll": "waistRoll",
        "waistPitch": "waistPitch",
        "torso": "torso",
        "shoulderPitchLeft": "shoulderPitchRight",
        "shoulderRollLeft": "shoulderRollRight",
        "shoulderYawLeft": "shoulderYawRight",
        "elbowLeft": "elbowRight",
        "wristYawLeft": "wristYawRight",
    }
)


def _make_mjlab_cfg():
    import mujoco
    from active_adaptation.assets.asset_cfg import AssetSpec, EntityCfg
    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg as MjlabContactSensorCfg
    from mjlab.utils.spec_config import CollisionCfg

    mjcf_path = ADAM_LITE_MJCF_PATH
    cfg = EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=ADAM_LITE_INIT_STATE.pos,
            joint_pos=ADAM_LITE_INIT_STATE.joint_pos,
            joint_vel=ADAM_LITE_INIT_STATE.joint_vel,
        ),
        spec_fn=lambda: mujoco.MjSpec.from_file(str(mjcf_path)),
        articulation=EntityArticulationInfoCfg(
            actuators=tuple(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(spec.joint_names_expr,),
                    effort_limit=spec.effort_limit,
                    stiffness=spec.stiffness,
                    damping=spec.damping,
                    armature=spec.armature,
                    frictionloss=spec.friction,
                )
                for spec in ADAM_LITE_ACTUATORS.values()
            )
        ),
        collisions=(
            CollisionCfg(
                geom_names_expr=(".*_collision",),
                contype=1,
                conaffinity=1,
                condim=3,
                priority=0,
                disable_other_geoms=False,
            ),
        ),
        joint_names_simulation=ADAM_LITE_JOINT_NAMES,
        body_names_simulation=ADAM_LITE_BODY_NAMES,
        joint_symmetry_mapping=JOINT_SYMMETRY_MAPPING,
        spatial_symmetry_mapping=SPATIAL_SYMMETRY_MAPPING,
    )
    sensors = (
        MjlabContactSensorCfg(
            name="contact_forces",
            primary=ContactMatch(
                mode="subtree",
                pattern=r"^(toeLeft|toeRight)$",
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain", entity=None),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            track_air_time=True,
            history_length=4,
        ),
        MjlabContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=4,
        ),
    )
    return AssetSpec(config=cfg, sensors=sensors)


def _make_isaaclab_cfg():
    cfg = AssetCfg(
        mjcf_path=ADAM_LITE_MJCF_PATH,
        usd_path=ADAM_LITE_URDF_PATH,
        init_state=ADAM_LITE_INIT_STATE,
        self_collisions=True,
        actuators=ADAM_LITE_ACTUATORS,
        joint_names_simulation=ADAM_LITE_JOINT_NAMES,
        body_names_simulation=ADAM_LITE_BODY_NAMES,
        joint_symmetry_mapping=JOINT_SYMMETRY_MAPPING,
        spatial_symmetry_mapping=SPATIAL_SYMMETRY_MAPPING,
    )
    return cfg.to_asset_spec("isaaclab")


def make_cfg(backend: str):
    if backend == "mjlab":
        return _make_mjlab_cfg()
    if backend == "isaaclab":
        return _make_isaaclab_cfg()
    raise ValueError(f"Unsupported backend for adam_lite_12dof: {backend}")


registry.register("asset", "adam_lite_12dof", make_cfg)

