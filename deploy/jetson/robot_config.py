"""Robot configuration for AAndrade-Humanoid-Lite deployment.

Architecture (2026-10-02): the Jetson talks Dynamixel Protocol 2.0 directly
via ROBOTIS's Python SDK, through an OpenRB-150 running ROBOTIS's STOCK
`usb_to_dynamixel` pass-through sketch. No custom firmware anywhere.

Every value here must match the TRAINED POLICY, not intuition. Ground truth
is policies/<run>/params/env.yaml from the training run.
"""

import glob

# ---------------------------------------------------------------------------
# JOINT ORDER — verified 2026-07-26 against the trained environment
# (env.unwrapped.scene['robot'].joint_names). PhysX orders joints
# breadth-first, which is why left/right interleave. Do not reorder.
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

# Dynamixel IDs — confirmed 2026-07-26, re-confirmed 2026-10-02 when all 14
# answered a ping (left leg 1-7, right leg 11-17, hip -> ankle down the chain).
SERVO_IDS = {
    "left_hip_pitch": 1,    "right_hip_pitch": 11,
    "left_hip_roll": 2,     "right_hip_roll": 12,
    "left_leg_yaw": 3,      "right_leg_yaw": 13,
    "left_knee1_pitch": 4,  "right_knee1_pitch": 14,
    "left_knee2_pitch": 5,  "right_knee2_pitch": 15,
    "left_ankle_pitch": 6,  "right_ankle_pitch": 16,
    "left_ankle_roll": 7,   "right_ankle_roll": 17,
}

# +1/-1 so a POSITIVE URDF angle moves the joint the same direction as in sim.
#
# STATUS: left +1 / right -1 is the mirrored-assembly assumption. It was
# CONFIRMED for the ankle_pitch pair by measurement on 2026-10-02 — resting
# ticks landed 274 below and 278 above centre respectively, and after the sign
# flip both reported -0.42 rad, i.e. the same physical pose. The other twelve
# were within 0.17 rad of zero, too close to call.
#
# Signs are still UNVERIFIED for those twelve. run_policy.py therefore clamps
# every target to DEFAULT +/- SAFE_EXCURSION and ramps in slowly, so a wrong
# sign shows up as drift you can kill rather than a joint slammed to its stop.
# Verify with:  python3 calibrate.py watch <joint_name>
SERVO_SIGNS = {
    "left_hip_pitch": -1,    "right_hip_pitch": -1,
    "left_hip_roll": +1,     "right_hip_roll": +1,
    "left_leg_yaw": +1,      "right_leg_yaw": +1,
    "left_knee1_pitch": -1,  "right_knee1_pitch": -1,
    "left_knee2_pitch": -1,  "right_knee2_pitch": -1,
    "left_ankle_pitch": -1,  "right_ankle_pitch": -1,
    "left_ankle_roll": +1,   "right_ankle_roll": +1,
}

# MEASURED 2026-10-02 with calibrate.py, robot held in the URDF zero pose
# (legs straight, feet flat, legs parallel). Sample spread was 0-1 ticks on
# every joint, so these are solid. Max deviation from a centred horn is 55
# ticks = 0.084 rad = 4.8 deg, which is ordinary assembly tolerance.
SERVO_OFFSETS = {
    "left_hip_pitch":    1993,
    "right_hip_pitch":   2040,
    "left_hip_roll":     2031,
    "right_hip_roll":    2067,
    "left_leg_yaw":      2088,
    "right_leg_yaw":     2051,
    "left_knee1_pitch":  2017,
    "right_knee1_pitch": 2010,
    "left_knee2_pitch":  2041,
    "right_knee2_pitch": 2035,
    "left_ankle_pitch":  2022,
    "right_ankle_pitch": 2044,
    "left_ankle_roll":   2008,
    "right_ankle_roll":  2020,
}

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

# --- SAFETY -----------------------------------------------------------------
# Hard clamp on how far any joint may be driven from its default pose, in rad.
# The policy's own output can reach +/- ACTION_SCALE * |action|, which is
# unbounded in principle. This caps the damage from a wrong sign, a bad
# observation, or an out-of-distribution policy output.
# 0.6 rad = 34 deg: enough for the trained crouch and gait, not enough to
# fold a leg backwards through a joint stop.
SAFE_EXCURSION = 0.6

# Seconds to ramp from wherever the robot is to DEFAULT_JOINT_POS when arming.
# Slow on purpose: a wrong sign becomes visible drift, not a slam.
RAMP_SECONDS = 4.0

# v1 was trained WITH base linear velocity in its observations (54 total).
# The robot cannot measure it, so v1 gets ZEROS there. v1 is a PIPELINE TEST
# on a stand, NOT expected to walk. The v2 retrain drops the term; set this
# False for v2 (observation size becomes 51).
INCLUDE_BASE_LIN_VEL = True

# Command limits (m/s, rad/s). Forward on this robot is -Y, established
# empirically during training. Keyboard 'w' maps to vy = -FWD_SPEED.
FWD_SPEED = 0.25
LAT_SPEED = 0.10
YAW_SPEED = 0.5

# ---------------------------------------------------------------------------
# IMU — BNO055 wired directly to the Jetson's 40-pin header
#   VIN -> pin 1 (3.3V) | GND -> pin 6 | SDA -> pin 3 | SCL -> pin 5
# Confirmed answering at 0x28 on bus 7. Mounting verified 2026-09-18:
# upright reads gravity (0.03, 0.02, -1.00), so no rotation needed.
# Driver runs in IMUPLUS mode: gyro + accel only, magnetometer ignored.
# ---------------------------------------------------------------------------
IMU_I2C_BUS = 7
IMU_I2C_ADDR = 0x28

IMU_TO_BASE = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]

IMU_MAX_CONSECUTIVE_FAILURES = 10

# ---------------------------------------------------------------------------
# Link — OpenRB-150 running stock usb_to_dynamixel, over USB CDC.
# The stock sketch calls dxl.begin(USB.baud()), so the servo bus baud
# automatically follows whatever we open the port at. No firmware config.
# ---------------------------------------------------------------------------
DXL_BAUDRATE = 1000000        # all XL330s set to 1 Mbps in July
POLICY_PATH = "~/aandrade-humanoid-isaaclab/policies/v1_walking/exported/policy.onnx"
CONTROL_HZ = 50.0


def find_serial_port():
    """First available CDC/FTDI port. The OpenRB moves between ttyACM0 and
    ttyACM1 across reflashes, so don't hardcode it."""
    for pattern in ("/dev/ttyACM*", "/dev/ttyUSB*"):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    return None


SERIAL_PORT = find_serial_port() or "/dev/ttyACM0"
