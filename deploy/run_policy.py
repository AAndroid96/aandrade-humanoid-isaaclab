#!/usr/bin/env python3
"""run_policy.py — Jetson-side brain for AAndrade-Humanoid-Lite.

50 Hz loop:
    read joint state over the Dynamixel bus  +  read BNO055 over local I2C
    -> build observation vector -> ONNX policy -> 14 joint targets -> bus

The OpenRB-150 runs ROBOTIS's stock `usb_to_dynamixel` pass-through sketch
and is a transparent wire. All protocol work happens here, via the official
Dynamixel SDK.

TWO MODES. Dry run is the default and cannot move the robot.

    python3 run_policy.py              # DRY RUN: reads, infers, PRINTS.
                                       # Torque is never enabled.
    python3 run_policy.py --arm        # Live. Enables torque. Robot moves.

Always dry-run first. It validates the observation vector, the IMU, the
joint readback and the policy output while the cost of a mistake is zero.

Live mode safety:
  * requires typing ARM at a prompt
  * writes goal = present position BEFORE enabling torque, so nothing jumps
  * ramps to the default crouch over RAMP_SECONDS instead of snapping
  * clamps every target to DEFAULT +/- SAFE_EXCURSION
  * watchdog kills torque if the IMU stops answering
  * 'k' kills torque instantly, ESC kills and exits

Keyboard teleop (terminal must have focus):
    w/s : forward / backward     (forward = -Y on this robot)
    a/d : strafe left / right
    q/e : turn left / right
  space : zero all commands
      k : KILL — torque off all servos immediately
    ESC : kill + exit

SAFETY: start with the robot ON A STAND, feet off the ground.
"""

import argparse
import os
import select
import sys
import termios
import time
import tty

import numpy as np
import onnxruntime as ort

import robot_config as cfg
from bno055_i2c import BNO055, BNO055Error
from dynamixel_bus import DynamixelBus

NUM_JOINTS = 14


def projected_gravity(quat_wxyz, imu_to_base: np.ndarray) -> np.ndarray:
    """World gravity direction (0,0,-1) expressed in the base frame."""
    w, x, y, z = quat_wxyz
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)
    g_imu = R.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return imu_to_base @ g_imu


class Keyboard:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *a):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def poll(self):
        keys = []
        while select.select([sys.stdin], [], [], 0)[0]:
            keys.append(sys.stdin.read(1))
        return keys


def build_calib():
    return [
        (cfg.SERVO_IDS[n], cfg.SERVO_SIGNS[n], cfg.SERVO_OFFSETS[n])
        for n in cfg.JOINT_NAMES
    ]


def ramp_to_default(bus, default_pos, seconds, hz=50.0):
    """Move from wherever the joints are to the default pose, slowly.

    Torque must already be on, with goal == present position, so this starts
    from a standstill. A wrong sign shows up here as a joint drifting the
    wrong way over several seconds — kill it with Ctrl+C.
    """
    start, _ = bus.read_state()
    start = np.array(start, dtype=np.float32)
    steps = max(1, int(seconds * hz))
    dt = 1.0 / hz

    print(f"Ramping to default pose over {seconds:.1f}s. "
          "Ctrl+C aborts and kills torque.")
    for i in range(1, steps + 1):
        alpha = i / steps
        target = (1.0 - alpha) * start + alpha * default_pos
        bus.write_targets(target)
        if i % int(hz / 2) == 0:
            print(f"  {100 * alpha:5.1f}%")
        time.sleep(dt)
    print("  ramp complete.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="store_true",
                    help="enable torque; the robot will move")
    ap.add_argument("--port", default=None,
                    help=f"serial port (default: autodetect, now "
                         f"{cfg.SERIAL_PORT})")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="dry run only: stop after N seconds (0 = forever)")
    args = ap.parse_args()

    live = args.arm
    port_name = args.port or cfg.SERIAL_PORT

    default_pos = np.array(
        [cfg.DEFAULT_JOINT_POS[n] for n in cfg.JOINT_NAMES], dtype=np.float32
    )
    imu_to_base = np.array(cfg.IMU_TO_BASE, dtype=np.float32)
    # Clamp to the INTERSECTION of how far we trust the policy and what the
    # joint can physically do. The knees are one-sided in the URDF, so a
    # symmetric band alone would permit unreachable angles.
    urdf_lo = np.array([cfg.JOINT_LIMITS[n][0] for n in cfg.JOINT_NAMES],
                       dtype=np.float32) + cfg.LIMIT_MARGIN
    urdf_hi = np.array([cfg.JOINT_LIMITS[n][1] for n in cfg.JOINT_NAMES],
                       dtype=np.float32) - cfg.LIMIT_MARGIN
    lo = np.maximum(default_pos - cfg.SAFE_EXCURSION, urdf_lo)
    hi = np.minimum(default_pos + cfg.SAFE_EXCURSION, urdf_hi)
    if np.any(lo > hi):
        bad = [n for i, n in enumerate(cfg.JOINT_NAMES) if lo[i] > hi[i]]
        sys.exit(f"Impossible clamp window for: {', '.join(bad)}\n"
                 "The default pose sits outside the URDF limits. Check "
                 "DEFAULT_JOINT_POS and JOINT_LIMITS.")

    mode = "LIVE — TORQUE WILL BE ENABLED" if live else "DRY RUN — nothing moves"
    print("=" * 68)
    print(f"  {mode}")
    print("=" * 68 + "\n")

    # --- policy ---
    policy_path = os.path.expanduser(cfg.POLICY_PATH)
    if not os.path.exists(policy_path):
        sys.exit(f"Policy not found at {policy_path}\n"
                 "Fix POLICY_PATH in robot_config.py.")
    sess = ort.InferenceSession(policy_path)
    in_name = sess.get_inputs()[0].name
    obs_dim = sess.get_inputs()[0].shape[-1]
    expected = 54 if cfg.INCLUDE_BASE_LIN_VEL else 51
    if obs_dim != expected:
        sys.exit(
            f"Policy expects obs dim {obs_dim} but config builds {expected}. "
            "Check INCLUDE_BASE_LIN_VEL in robot_config.py."
        )
    print(f"Policy loaded ({obs_dim} obs -> {NUM_JOINTS} actions).")
    if cfg.INCLUDE_BASE_LIN_VEL:
        print("NOTE: this policy expects base linear velocity, which the "
              "robot cannot\n      measure. Those 3 inputs are ZEROS. "
              "Expect pose-holding, not walking.")

    # --- IMU ---
    try:
        imu = BNO055(bus=cfg.IMU_I2C_BUS, address=cfg.IMU_I2C_ADDR)
    except (BNO055Error, OSError) as exc:
        sys.exit(f"BNO055 not available: {exc}")
    quat, gyro = imu.read_state()
    imu_failures = 0
    print(f"IMU online (IMUPLUS). calib sys/gyro/accel/mag = "
          f"{imu.calibration()}")

    # --- servo bus ---
    try:
        bus = DynamixelBus(port_name, cfg.DXL_BAUDRATE, build_calib())
    except RuntimeError as exc:
        imu.close()
        sys.exit(f"Dynamixel bus: {exc}")

    found = bus.ping_all()
    if len(found) != NUM_JOINTS:
        missing = [n for n in cfg.JOINT_NAMES if cfg.SERVO_IDS[n] not in found]
        bus.close()
        imu.close()
        sys.exit(
            f"Only {len(found)}/{NUM_JOINTS} servos answered.\n"
            f"Missing: {', '.join(missing)}\n"
            "Not starting with an incomplete robot."
        )
    print(f"All {NUM_JOINTS} servos present on {port_name}.\n")

    # --- arming ---
    if live:
        print("!" * 68)
        print("  Robot ON A STAND, feet clear of the ground?")
        print("  Hand on the power switch?")
        print(f"  Targets clamped to default +/- {cfg.SAFE_EXCURSION} rad,")
        print(f"  further limited to the URDF joint ranges "
              f"(margin {cfg.LIMIT_MARGIN} rad).")
        print("!" * 68)
        if input('\nType ARM to enable torque, anything else to abort: ') \
                .strip() != "ARM":
            bus.close()
            imu.close()
            sys.exit("Aborted. Torque was never enabled.")

        bus.configure_position_mode()          # leaves torque OFF
        present, _ = bus.read_state()
        bus.write_targets(present)             # goal = here, so no jump
        bus.set_torque(True)
        print("\nTorque ON, holding present position.")
        try:
            ramp_to_default(bus, default_pos, cfg.RAMP_SECONDS, cfg.CONTROL_HZ)
        except KeyboardInterrupt:
            bus.set_torque(False)
            bus.close()
            imu.close()
            sys.exit("\nRamp aborted — torque off.")
        input("\nPress Enter to hand control to the policy "
              "(Ctrl+C to stop here)... ")

    prev_action = np.zeros(NUM_JOINTS, dtype=np.float32)
    command = np.zeros(3, dtype=np.float32)   # vx, vy, wz
    dt = 1.0 / cfg.CONTROL_HZ

    if live:
        print("\nRunning. w/s a/d q/e drive, space=stop, k=KILL, ESC=exit.\n")
    else:
        print("Dry run. Ctrl+C to stop. Printing 2x per second.\n")

    loops = 0
    clamp_hits = 0
    t_start = time.monotonic()

    with Keyboard() as kb:
        next_tick = time.monotonic()
        try:
            while True:
                for key in kb.poll():
                    if key == "w":   command[1] = -cfg.FWD_SPEED  # forward=-Y
                    elif key == "s": command[1] = +cfg.FWD_SPEED
                    elif key == "a": command[0] = +cfg.LAT_SPEED
                    elif key == "d": command[0] = -cfg.LAT_SPEED
                    elif key == "q": command[2] = +cfg.YAW_SPEED
                    elif key == "e": command[2] = -cfg.YAW_SPEED
                    elif key == " ": command[:] = 0.0
                    elif key == "k" and live:
                        bus.set_torque(False)
                        print("\nKILL — servos limp. Restart to re-arm.")
                        return
                    elif key == "\x1b":                            # ESC
                        if live:
                            bus.set_torque(False)
                        print("\nExit.")
                        return

                # --- joint state ---
                joint_pos, joint_vel = bus.read_state()
                joint_pos = np.array(joint_pos, dtype=np.float32)
                joint_vel = np.array(joint_vel, dtype=np.float32)

                # --- orientation ---
                try:
                    quat, gyro = imu.read_state()
                    imu_failures = 0
                except OSError:
                    imu_failures += 1
                    if imu_failures >= cfg.IMU_MAX_CONSECUTIVE_FAILURES:
                        if live:
                            bus.set_torque(False)
                        print(f"\nIMU stopped responding ({imu_failures} "
                              "cycles) — stopping.")
                        return

                base_ang_vel = imu_to_base @ np.array(gyro, dtype=np.float32)
                grav = projected_gravity(quat, imu_to_base)

                # --- observation (order must match training) ---
                parts = []
                if cfg.INCLUDE_BASE_LIN_VEL:
                    parts.append(np.zeros(3, dtype=np.float32))  # unmeasurable
                parts += [
                    base_ang_vel,
                    grav,
                    command,
                    joint_pos - default_pos,
                    joint_vel,
                    prev_action,
                ]
                obs = np.concatenate(parts)[None, :].astype(np.float32)

                # --- inference ---
                action = sess.run(None, {in_name: obs})[0][0].astype(np.float32)
                prev_action = action
                raw_targets = default_pos + cfg.ACTION_SCALE * action
                targets = np.clip(raw_targets, lo, hi)
                if not np.allclose(targets, raw_targets, atol=1e-6):
                    clamp_hits += 1

                if live:
                    bus.write_targets(targets)

                loops += 1

                # --- reporting ---
                if not live and loops % 25 == 0:
                    el = time.monotonic() - t_start
                    print(f"[{el:6.1f}s  {loops/el:4.1f} Hz  "
                          f"clamped {clamp_hits}]")
                    print(f"  grav      {grav[0]:+.2f} {grav[1]:+.2f} "
                          f"{grav[2]:+.2f}   (upright ~ 0, 0, -1)")
                    print(f"  gyro      {base_ang_vel[0]:+.2f} "
                          f"{base_ang_vel[1]:+.2f} {base_ang_vel[2]:+.2f} rad/s")
                    print(f"  cmd       {command[0]:+.2f} {command[1]:+.2f} "
                          f"{command[2]:+.2f}")
                    print(f"  |action|  max {np.abs(action).max():.2f}  "
                          f"mean {np.abs(action).mean():.2f}")
                    print("  target vs present (rad):")
                    for i, n in enumerate(cfg.JOINT_NAMES):
                        mark = " CLAMP" if abs(
                            targets[i] - raw_targets[i]) > 1e-6 else ""
                        print(f"    {n:<20} tgt {targets[i]:+.3f}  "
                              f"now {joint_pos[i]:+.3f}{mark}")
                    print()

                if not live and args.seconds and \
                        time.monotonic() - t_start >= args.seconds:
                    print("Dry run time limit reached.")
                    return

                # --- 50 Hz pacing ---
                next_tick += dt
                sleep = next_tick - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_tick = time.monotonic()   # fell behind: resync
        finally:
            if live:
                bus.set_torque(False)
            bus.close()
            imu.close()
            el = max(1e-6, time.monotonic() - t_start)
            print(f"\nStopped. {loops} cycles, {loops/el:.1f} Hz average, "
                  f"{clamp_hits} clamped.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
