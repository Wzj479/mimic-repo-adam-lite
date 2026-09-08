# Adam Lite 12DOF Humanoid Robot Training Environment

This directory contains the reinforcement learning training environment for the **Adam Lite 12DOF** humanoid robot, modeled after the adam_lite implementation pattern.

## Robot Overview

Adam Lite is a humanoid robot with:
- **12 degrees of freedom (DOF)**: 6 DOF per leg
- **Fixed upper body**: Arms and torso joints are fixed
- **Mass**: ~10.36 kg (pelvis) + additional link masses
- **Standing height**: ~0.85m (approximate)

### Joint Configuration

#### Left Leg (6 DOF):
1. `hipPitch_Left` - Hip pitch joint (±2.164 rad, 230 Nm)
2. `hipRoll_Left` - Hip roll joint (-0.733 to 1.605 rad, 160 Nm)
3. `hipYaw_Left` - Hip yaw joint (±0.785 rad, 105 Nm)
4. `kneePitch_Left` - Knee pitch joint (0.052 to 2.391 rad, 230 Nm)
5. `anklePitch_Left` - Ankle pitch joint (-1.0 to 0.35 rad, 40 Nm)
6. `ankleRoll_Left` - Ankle roll joint (±0.3491 rad, 12 Nm)

#### Right Leg (6 DOF):
7. `hipPitch_Right` - Hip pitch joint (±2.164 rad, 230 Nm)
8. `hipRoll_Right` - Hip roll joint (-1.605 to 0.733 rad, 160 Nm)
9. `hipYaw_Right` - Hip yaw joint (±0.785 rad, 105 Nm)
10. `kneePitch_Right` - Knee pitch joint (0.052 to 2.391 rad, 230 Nm)
11. `anklePitch_Right` - Ankle pitch joint (-1.0 to 0.35 rad, 40 Nm)
12. `ankleRoll_Right` - Ankle roll joint (±0.3491 rad, 12 Nm)

## Files

- `adam_lite_config.py` - Configuration file containing robot parameters, rewards, and training settings
- `adam_lite_env.py` - Environment implementation with custom reward functions and observations
- `README.md` - This documentation file

## Configuration Details

### Control Parameters

The environment uses Position (P) control with the following PD gains:

```python
stiffness = {
    'hipPitch': 120,    # N*m/rad
    'hipRoll': 80,      # N*m/rad
    'hipYaw': 60,       # N*m/rad
    'kneePitch': 120,   # N*m/rad
    'anklePitch': 25,   # N*m/rad
    'ankleRoll': 8,     # N*m/rad
}

damping = {
    'hipPitch': 2.5,    # N*m*s/rad
    'hipRoll': 2.0,     # N*m*s/rad
    'hipYaw': 1.5,      # N*m*s/rad
    'kneePitch': 3.0,   # N*m*s/rad
    'anklePitch': 1.5,  # N*m*s/rad
    'ankleRoll': 1.0,   # N*m*s/rad
}
```

- **Action scale**: 0.25
- **Control decimation**: 4 (control frequency = 50 Hz at dt=0.005s)

### Default Joint Angles

Standing pose (all values in radians):
```python
default_joint_angles = {
    'hipPitch_Left': -0.1,
    'hipRoll_Left': 0,
    'hipYaw_Left': 0.0,
    'kneePitch_Left': 0.3,
    'anklePitch_Left': -0.2,
    'ankleRoll_Left': 0,
    'hipPitch_Right': -0.1,
    'hipRoll_Right': 0,
    'hipYaw_Right': 0.0,
    'kneePitch_Right': 0.3,
    'anklePitch_Right': -0.2,
    'ankleRoll_Right': 0,
}
```

### Observation Space (47 dimensions)

1. **Base angular velocity** (3): Angular velocity in body frame
2. **Projected gravity** (3): Gravity vector in body frame
3. **Commands** (3): Linear velocity x, y and angular velocity z commands
4. **Joint positions** (12): Current joint positions minus default positions
5. **Joint velocities** (12): Current joint velocities
6. **Previous actions** (12): Previous action commands
7. **Phase information** (2): sin(phase) and cos(phase) for gait timing

### Privileged Observation Space (50 dimensions)

Same as observation space plus:
- **Base linear velocity** (3): Linear velocity in body frame

### Reward Functions

The environment includes the following reward components:

1. **tracking_lin_vel** (1.0): Rewards tracking linear velocity commands
2. **tracking_ang_vel** (0.5): Rewards tracking angular velocity commands
3. **lin_vel_z** (-2.0): Penalizes vertical velocity
4. **ang_vel_xy** (-0.05): Penalizes roll and pitch angular velocities
5. **orientation** (-1.0): Penalizes non-upright orientation
6. **base_height** (-10.0): Penalizes deviation from target height (0.80m)
7. **dof_acc** (-2.5e-7): Penalizes joint accelerations
8. **dof_vel** (-1e-3): Penalizes joint velocities
9. **action_rate** (-0.01): Penalizes action changes
10. **dof_pos_limits** (-5.0): Penalizes approaching joint limits
11. **alive** (0.15): Rewards staying alive
12. **hip_pos** (-1.0): Penalizes hip roll and yaw deviations
13. **contact_no_vel** (-0.2): Penalizes foot contact with velocity
14. **feet_swing_height** (-20.0): Rewards proper swing height (0.08m)
15. **contact** (0.18): Rewards proper contact timing with gait phase

### Domain Randomization

- **Friction randomization**: Range [0.1, 1.25]
- **Base mass randomization**: Added mass range [-1.0, 3.0] kg
- **Push robots**: Random pushes every 5 seconds, max velocity 1.5 m/s

## Training

### Start Training

```bash
cd /path/to/pnd_rl_gym
python legged_gym/scripts/train.py --task=adam_lite_12dof
```

### Training Parameters

- **Policy network**: Actor-Critic with LSTM (Recurrent)
- **Actor hidden dimensions**: [32]
- **Critic hidden dimensions**: [32]
- **LSTM hidden size**: 64
- **LSTM layers**: 1
- **Activation**: ELU
- **Initial noise std**: 0.8
- **Entropy coefficient**: 0.01
- **Max iterations**: 10,000

### WandB Integration

The environment is configured to use Weights & Biases (WandB) for experiment tracking:
- **Project name**: `pndbotics_adam_lite_12dof_locomotion`
- **Tags**: `['adam_lite_12dof', 'rough_terrain', 'lstm']`

## Testing

### Play Trained Policy

```bash
python legged_gym/scripts/play.py --task=adam_lite_12dof
```

## Deployment

### MuJoCo Simulation

Configuration file: `deploy/deploy_mujoco/configs/adam_lite_12dof.yaml`

```bash
cd deploy/deploy_mujoco
python deploy_mujoco.py --config configs/adam_lite_12dof.yaml
```

### Real Robot

Configuration file: `deploy/deploy_real/configs/adam_lite_12dof.yaml`

```bash
cd deploy/deploy_real
python deploy_real.py --config configs/adam_lite_12dof.yaml
```

## Environment Implementation Details

### Custom Reward Functions

The environment implements several custom reward functions:

#### `_reward_contact()`
Rewards proper foot contact timing based on gait phase. Stance phase is defined as phase < 0.55.

#### `_reward_feet_swing_height()`
Encourages feet to reach a target height of 0.08m during swing phase.

#### `_reward_contact_no_vel()`
Penalizes feet that are in contact but still moving (should be stationary during stance).

#### `_reward_hip_pos()`
Penalizes deviations in hip roll and hip yaw joints from their default positions to encourage stable walking.

### Gait Phase Tracking

The environment tracks gait phases for each leg:
- **Period**: 0.8 seconds
- **Phase offset**: 0.5 (legs are 180° out of phase)
- **Stance phase**: phase < 0.55
- **Swing phase**: phase >= 0.55

## Notes

1. The upper body (torso, arms) joints are **fixed** in this configuration
2. The robot is designed for bipedal locomotion research
3. Foot bodies are named `toeLeft` and `toeRight`
4. Base link is `pelvis`, which contains the IMU
5. The environment follows the same structure as adam_lite robots for consistency

## References

- Base implementation: `legged_gym/envs/base/legged_robot.py`
- Similar environments: adam_lite
- URDF file: `resources/robots/adam_lite_12dof/adam_lite_12dof.urdf`

## Troubleshooting

1. **Robot falls immediately**: 
   - Check initial height in config (`pos = [0.0, 0.0, 0.85]`)
   - Verify PD gains are appropriate for the robot mass

2. **Poor tracking performance**:
   - Adjust reward scales in config
   - Increase training iterations
   - Check command velocities are reasonable

3. **Unstable gait**:
   - Tune `contact` and `feet_swing_height` reward scales
   - Adjust gait period and phase offset
   - Increase `hip_pos` penalty scale

## Future Improvements

- [ ] Add upper body joint control (arms and torso)
- [ ] Implement terrain curriculum
- [ ] Add vision-based observations
- [ ] Support for manipulation tasks
- [ ] Multi-modal locomotion (walking, running, jumping)

