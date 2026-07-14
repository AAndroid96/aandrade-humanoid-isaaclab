"""Flat-terrain velocity-tracking env for AAndrade-Humanoid-Lite.

Follows the same pattern as the Unitree H1/G1 configs in
isaaclab_tasks.manager_based.locomotion.velocity — inherit the template,
swap the robot, rescale everything for a ~0.7 kg, ~25 cm-tall biped.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

from aandrade_humanoid.robots.aandrade_humanoid import AANDRADE_HUMANOID_CFG
from aandrade_humanoid.tasks.velocity import custom_rewards

# Link names from the URDF (onshape-to-robot naming)
PELVIS = "dc15_a01_adaptor_2_1"
FEET = ["servobody_4", "servobody_8"]          # left, right foot
UPPER_LEG_BODIES = ["servohorn.*", "servobody", "servobody_2", "servobody_3",
                    "servobody_5", "servobody_6", "servobody_7"]


@configclass
class AAndradeHumanoidFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------------------
        # Scene / robot
        # ------------------------------------------------------------------
        self.scene.robot = AANDRADE_HUMANOID_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        # Flat terrain, no height scanner
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # ------------------------------------------------------------------
        # Commands — scaled for a mini biped (H1 walks at 1 m/s; this won't)
        # ------------------------------------------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-0.10, 0.10)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.30, 0.10)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)

        # ------------------------------------------------------------------
        # Events — gentler pushes and mass randomization for a 0.7 kg robot
        # ------------------------------------------------------------------
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.15, 0.15), "y": (-0.15, 0.15)
        }
        self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=PELVIS
        )
        # Newer isaaclab_tasks versions add a base CoM randomization event
        # that defaults to a body named "base" — retarget it to our pelvis.
        if hasattr(self.events, "base_com") and self.events.base_com is not None:
            self.events.base_com.params["asset_cfg"] = SceneEntityCfg(
                "robot", body_names=PELVIS
            )
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.03, 0.05)
        self.events.base_external_force_torque.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=PELVIS
        )
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-3.14, 3.14)},
            "velocity_range": {k: (0.0, 0.0) for k in
                               ("x", "y", "z", "roll", "pitch", "yaw")},
        }

        # ------------------------------------------------------------------
        # Rewards — retargeted body names + small-robot scales
        # ------------------------------------------------------------------
        self.rewards.feet_air_time = RewTerm(
            func=mdp.feet_air_time_positive_biped,
            weight=2.0,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
                "threshold": 0.3,
            },
        )
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=UPPER_LEG_BODIES
                ),
                "threshold": 1.0,
            },
        )
        self.rewards.flat_orientation_l2.weight = -2.5
        # --- gait shaping (anti-scoot, anti-leg-twist) ---
        # Penalize feet sliding while in contact: kills the skating gait.
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.5,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
                "asset_cfg": SceneEntityCfg("robot", body_names=FEET),
            },
        )
        # Pay for actually lifting feet during swing: anti-ankle-shuffle.
        self.rewards.feet_clearance = RewTerm(
            func=custom_rewards.feet_clearance,
            weight=0.75,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
                "asset_cfg": SceneEntityCfg("robot", body_names=FEET),
                "target_height": 0.09,
            },
        )
        # Keep the sole parallel to the ground through the whole stride.
        self.rewards.feet_flat = RewTerm(
            func=custom_rewards.feet_flat,
            weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        # Split knee flexion across both tandem knee servos.
        self.rewards.knee_tandem_share = RewTerm(
            func=custom_rewards.knee_tandem_share,
            weight=-0.5,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        # Anchor knees near the crouched default: straight-leg walking
        # (0.35 rad deviation on every knee) costs steadily, oscillating
        # around the bend is cheap -> keeps knees engaged in the gait.
        self.rewards.knee_anchor = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.5,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=[".*_knee1_pitch", ".*_knee2_pitch"]
                )
            },
        )
        # A planted foot gets expensive after 0.4s -> forces alternating steps
        # (kills the one-legged vaulting/limp local optimum).
        self.rewards.feet_stance_time = RewTerm(
            func=custom_rewards.feet_stance_time,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
                "command_name": "base_velocity",
                "threshold": 0.4,
            },
        )
        # Keep legs in the sagittal plane: penalize yaw/roll joint deviation
        # from default so stepping is done with the pitch joints.
        self.rewards.joint_deviation_yaw = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.3,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_yaw"])},
        )
        self.rewards.joint_deviation_roll = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.15,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=[".*_hip_roll", ".*_ankle_roll"]
                )
            },
        )
        self.rewards.dof_torques_l2.weight = -2.0e-3   # torques are ~0.1-0.5 N.m here
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.lin_vel_z_l2.weight = -1.0

        # ------------------------------------------------------------------
        # Terminations — fall = pelvis contact
        # ------------------------------------------------------------------
        self.terminations.base_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=[PELVIS, "servohorn", "servohorn_4"]
                ),
                "threshold": 0.5,   # N — scaled down from 1.0 for a light robot
            },
        )


@configclass
class AAndradeHumanoidFlatEnvCfg_PLAY(AAndradeHumanoidFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_external_force_torque = None

