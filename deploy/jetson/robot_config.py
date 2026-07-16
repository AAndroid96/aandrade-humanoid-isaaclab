"""Robot configuration for AAndrade-Humanoid-Lite deployment.

Every value here must match the TRAINED POLICY, not intuition.
The ground truth is policies/<run>/params/env.yaml from the training run.
"""

# ---------------------------------------------------------------------------
# JOINT ORDER — must match the simulator's joint ordering EXACTLY.
#
# !!! VERIFY BEFORE FIRST RUN !!!
# PhysX orders joints breadth-first over the articulation tree, NOT in URDF
# file order. The list below is the expected BFS order for your robot, but
# confirm it by running this on your DESKTOP (in ~/isaaclab):
#
#   ./isaaclab.sh -p -c "
#   from isaaclab.app import AppLauncher
#   app = AppLauncher(headless=True).app
#   import gymnasium as gym
#   import aandrade_humanoid.tasks
#   from isaaclab_tasks.utils import parse_env_cfg
#   cfg = parse_env_cfg('Isaac-Velocity-Flat-AAndrade-Humanoid-v0', num_envs=1)
#   env = gym.make('Isaac-Velocity-Flat-AAndrade-Humanoid-v0', cfg=cfg)
#   print(env.unwrapped.scene['robot'].joint_names)"
#
# Paste the printed list here verbatim. The firmware's CALIB table uses the
# same index order.
# ---------------------------------------------------------------------------
JOINT_NAMES = [
    "left_hip_pitch",
    "right_hip_pitch",
    "left_hip_roll",
    "right_hip_roll",
    "left_leg_yaw",
    "right_leg_yaw",
    "left_knee1_pitch",
    "right_knee1_pitch",
    "left_knee2_pitch",
    "right_knee2_pitch",
    "left_ankle_pitch",
    "right_ankle_pitch",
    "left_ankle_roll",
    "right_ankle_roll",
]

# Crouched default pose (rad) — from robots/aandrade_humanoid.py.
# Observations use joint_pos RELATIVE to these; actions are offsets FROM these.
DEFAULT_JOINT_POS = {
    "left_hip_pitch": 0.35,
    "right_hip_pitch": -0.35,
    "left_knee1_pitch": 0.35,
    "right_knee1_pitch": -0.35,
    "left_knee2_pitch": -0.35,
    "right_knee2_pitch": 0.35,
    "left_ankle_pitch": -0.35,
    "right_ankle_pitch": 0.35,
    "left_hip_roll": 0.0,
    "right_hip_roll": 0.0,
    "left_leg_yaw": 0.0,
    "right_leg_yaw": 0.0,
    "left_ankle_roll": 0.0,
    "right_ankle_roll": 0.0,
}

# target_joint_pos = default + ACTION_SCALE * policy_action
# Verify against params/env.yaml -> actions -> joint_pos -> scale
ACTION_SCALE = 0.5

# v1 policy was trained WITH base linear velocity in observations (54 total).
# The real robot cannot measure it, so v1 feeds ZEROS there — enough for
# plumbing tests on a stand, NOT expected to walk. The v2 retrain drops the
# term; set this to False for v2 (observation size becomes 51).
INCLUDE_BASE_LIN_VEL = True

# Command limits (m/s, rad/s). Forward on this robot is -Y (established
# empirically during training). Keyboard teleop maps W -> vy = -FWD_SPEED.
FWD_SPEED = 0.25
LAT_SPEED = 0.10
YAW_SPEED = 0.5

# IMU mounting rotation: rotates BNO055 body-frame vectors into the robot
# BASE frame. Identity assumes the BNO055 axes are aligned with the base
# frame axes. If your IMU is mounted rotated, put the 3x3 rotation here.
# Verify: with the robot upright and still, projected gravity must print
# close to (0, 0, -1); tilt the robot forward (-Y) and the -Y component
# should grow.
IMU_TO_BASE = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]

SERIAL_PORT = "/dev/ttyACM0"
POLICY_PATH = "~/aandrade-humanoid-isaaclab/policies/v1_walking/exported/policy.onnx"
CONTROL_HZ = 50.0
