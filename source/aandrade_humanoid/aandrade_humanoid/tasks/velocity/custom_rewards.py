"""Custom reward terms for AAndrade-Humanoid-Lite gait shaping."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_clearance(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.05,
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward swing-foot height, gated so it can't be farmed by falling over.

    A foot earns clearance reward only when ALL of:
      1. that foot is in swing (not in contact),
      2. exactly one foot is in contact (single support = a real step),
      3. the robot is upright (gravity projects mostly down its body z).

    Gates 2 and 3 close the "lie on the floor and wave your legs" exploit.
    """
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    in_contact = net_forces.norm(dim=-1).max(dim=1)[0] > 1.0  # (envs, feet)
    # Gate 2: single support — exactly one foot planted.
    single_support = in_contact.sum(dim=-1) == 1
    # Gate 3: upright — projected gravity mostly along -z of the base.
    upright = asset.data.projected_gravity_b[:, 2] < -0.8
    # Swing-foot height, saturated at target.
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    clearance = torch.clamp(foot_height / target_height, max=1.0)
    reward = (clearance * (~in_contact).float()).sum(dim=1)
    # Gate 4 (v3): only pay while the robot is COMMANDED to move. Without
    # this, lifting a foot is profitable at zero command, which makes
    # marching in place strictly better than standing still.
    cmd = env.command_manager.get_command(command_name)
    moving = cmd[:, :3].norm(dim=1) > command_threshold
    return reward * single_support.float() * upright.float() * moving.float()


def feet_flat(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize foot tilt via the planar-chain identity (AAndrade-specific).

    With all pitch axes parallel to base X, the foot's tilt relative to the
    base equals the signed sum of the leg's pitch joints. Axis signs from
    the URDF: left = -hip + k1 - k2 + ankle, right mirrored. Zero sum =
    sole parallel to the ground (given a level base, which
    flat_orientation_l2 already enforces).
    """
    asset = env.scene[asset_cfg.name]
    names = [
        "left_hip_pitch", "left_knee1_pitch", "left_knee2_pitch", "left_ankle_pitch",
        "right_hip_pitch", "right_knee1_pitch", "right_knee2_pitch", "right_ankle_pitch",
    ]
    ids, _ = asset.find_joints(names, preserve_order=True)
    jp = asset.data.joint_pos[:, ids]
    left_tilt = -jp[:, 0] + jp[:, 1] - jp[:, 2] + jp[:, 3]
    right_tilt = jp[:, 4] - jp[:, 5] + jp[:, 6] - jp[:, 7]
    return left_tilt.square() + right_tilt.square()


def knee_tandem_share(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize unequal use of the tandem knee pair.

    The two knee servos flex the same physical direction with opposite
    signs (left k1 >= 0, k2 <= 0). Equal sharing means k1 + k2 == 0, so
    the squared sum penalizes dumping all flexion on one servo.
    """
    asset = env.scene[asset_cfg.name]
    names = [
        "left_knee1_pitch", "left_knee2_pitch",
        "right_knee1_pitch", "right_knee2_pitch",
    ]
    ids, _ = asset.find_joints(names, preserve_order=True)
    jp = asset.data.joint_pos[:, ids]
    left_imbalance = (jp[:, 0] + jp[:, 1]).square()
    right_imbalance = (jp[:, 2] + jp[:, 3]).square()
    return left_imbalance + right_imbalance


def feet_stance_time(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    threshold: float = 0.4,
) -> torch.Tensor:
    """Penalize a foot staying planted longer than ``threshold`` seconds.

    Kills the vaulting/limping local optimum (one rigid stance leg, one
    working leg) by making a locked stance foot increasingly costly, which
    forces alternating steps. Only active when the robot is commanded to
    move, so standing still at ~zero command remains free.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    excess = (contact_time - threshold).clip(min=0.0, max=1.0).sum(dim=1)
    cmd = env.command_manager.get_command(command_name)
    moving = cmd[:, :2].norm(dim=1) > 0.05
    return excess * moving.float()


# --------------------------------------------------------------------------
# STAND-STILL (v3)
# --------------------------------------------------------------------------
def stand_still(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize joint deviation from the default pose while commanded to stand.

    Returns 0 whenever a velocity or yaw command is present, so this never
    fights the gait. At zero command it pulls every joint back toward the
    trained crouch, which is what "stand in place" means for this robot.

    Keep the weight modest: too strong and it overrides balance recovery,
    because catching a push REQUIRES moving away from the default pose.
    """
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    standing = cmd[:, :3].norm(dim=1) < command_threshold
    deviation = (asset.data.joint_pos - asset.data.default_joint_pos).abs().sum(dim=1)
    return deviation * standing.float()
