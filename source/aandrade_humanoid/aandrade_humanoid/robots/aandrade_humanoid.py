"""Articulation config for the AAndrade-Humanoid-Lite (14-DOF, XL330-M288-T).

Round 5: crouched nominal pose. Bent knees make knee-driven gait the
cheapest motion for the policy. Angles are geometry-verified from the URDF
axis directions so the feet sit flat and the pelvis stays centered.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

AANDRADE_USD_PATH = os.environ.get(
    "AANDRADE_USD_PATH",
    os.path.expanduser("~/aandrade_isaaclab/robot_isaac.usd"),
)

AANDRADE_HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=AANDRADE_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.30),
        joint_pos={
            # CROUCHED default pose, geometry-verified for flat feet:
            # per-leg pitch rotations sum to zero foot tilt, pelvis centered.
            # ~20 deg thigh/shank tilt = bent-knee nominal, so the policy's
            # cheapest motions are knee oscillations (human-like gait basis).
            "left_hip_pitch": 0.35,
            "right_hip_pitch": -0.35,
            "left_knee1_pitch": 0.35,
            "right_knee1_pitch": -0.35,
            "left_knee2_pitch": -0.35,
            "right_knee2_pitch": 0.35,
            "left_ankle_pitch": -0.35,
            "right_ankle_pitch": 0.35,
            ".*_hip_roll": 0.0,
            ".*_leg_yaw": 0.0,
            ".*_ankle_roll": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*"],
            effort_limit=0.45,
            saturation_effort=0.52,
            velocity_limit=10.5,
            stiffness=4.0,
            damping=0.15,
            armature=0.002,
            friction=0.01,
        ),
    },
)

