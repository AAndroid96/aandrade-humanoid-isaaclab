#!/usr/bin/env python3
"""hold_pose.py -- first-torque validation WITHOUT the policy in the loop.

Why this exists: validating hardware with a neural network in the loop means
two unknowns at once. This commands fixed, known poses instead, so torque,
offsets, signs and mechanical range get tested one variable at a time.

MODES

  one <joint> [delta]   Torque ONLY that one servo and move it `delta` radians
                        from where it is (default 0.25, hard-capped 0.40).
                        The other thirteen stay limp. This is the safest
                        possible powered test: one joint, small move, bounded.
                        Use it to verify SIGNS.

  crouch                Ramp all fourteen to the trained default crouch and
                        hold it. Use it to check the pose looks right and
                        that the servos can hold it.

  zero                  Ramp all fourteen to URDF zero (straight legs).

IMPORTANT -- WHY YOU MUST LOOK AT THE ROBOT

A wrong sign is invisible in software. Reads and writes both go through the
same sign, so the error cancels: command +0.3, the servo moves to mirrored
-0.3 physically, and readback reports +0.3. Tracking error will be zero
while the joint is in the wrong place.

The ONLY way to catch a wrong sign is to watch the hardware and compare
against what the pose should look like. This script tells you what to expect;
your eyes are the instrument.

Usage:
    python3 hold_pose.py one left_knee1_pitch
    python3 hold_pose.py one right_hip_pitch 0.2
    python3 hold_pose.py crouch
    python3 hold_pose.py zero
"""

import argparse
import sys
import time

import numpy as np

import robot_config as cfg
from dynamixel_bus import ADDR_TORQUE_ENABLE, DynamixelBus

MAX_DELTA = 0.40          # rad, hard cap for single-joint moves
SETTLE_S = 1.5


def build_calib():
    return [
        (cfg.SERVO_IDS[n], cfg.SERVO_SIGNS[n], cfg.SERVO_OFFSETS[n])
        for n in cfg.JOINT_NAMES
    ]


def open_bus():
    try:
        bus = DynamixelBus(cfg.SERIAL_PORT, cfg.DXL_BAUDRATE, build_calib())
    except RuntimeError as exc:
        sys.exit(f"Dynamixel bus: {exc}")
    found = bus.ping_all()
    if len(found) != 14:
        missing = [n for n in cfg.JOINT_NAMES if cfg.SERVO_IDS[n] not in found]
        bus.close()
        sys.exit(f"Only {len(found)}/14 answered. Missing: {', '.join(missing)}")
    print(f"All 14 servos present on {cfg.SERIAL_PORT}.")
    return bus


def torque_one(bus, dxl_id, on):
    """Torque a single servo, leaving the others untouched."""
    bus.packet.write1ByteTxRx(bus.port, dxl_id, ADDR_TORQUE_ENABLE,
                              1 if on else 0)


def confirm(prompt):
    return input(f"\n{prompt}\nType YES to continue, anything else to "
                 "abort: ").strip() == "YES"


def cmd_one(bus, joint, delta):
    if joint not in cfg.SERVO_IDS:
        bus.close()
        print(f"Unknown joint '{joint}'. Valid names:")
        for n in cfg.JOINT_NAMES:
            print(f"  {n}")
        return

    delta = float(np.clip(delta, -MAX_DELTA, MAX_DELTA))
    idx = cfg.JOINT_NAMES.index(joint)
    dxl_id = cfg.SERVO_IDS[joint]
    sign = cfg.SERVO_SIGNS[joint]

    pos, _ = bus.read_state()
    start = float(pos[idx])
    target = start + delta

    print(f"\n  joint   {joint}  (id {dxl_id}, sign {sign:+d})")
    print(f"  now     {start:+.3f} rad")
    print(f"  target  {target:+.3f} rad   (delta {delta:+.3f})")
    print(f"\n  The other 13 servos stay LIMP. Only this one gets torque.")

    if not confirm("Robot supported, hand near the power switch?"):
        bus.close()
        sys.exit("Aborted. No torque was enabled.")

    # Goal = present for every joint, so enabling torque cannot cause a jump.
    bus.write_targets(pos)
    torque_one(bus, dxl_id, True)
    print("\n  torque on, holding present position...")
    time.sleep(0.4)

    steps = 40
    try:
        for i in range(1, steps + 1):
            goal = list(pos)
            goal[idx] = start + delta * (i / steps)
            bus.write_targets(goal)
            time.sleep(0.03)
    except KeyboardInterrupt:
        torque_one(bus, dxl_id, False)
        bus.close()
        sys.exit("\nAborted mid-move. Torque off.")

    time.sleep(SETTLE_S)
    pos2, _ = bus.read_state()
    reached = float(pos2[idx])
    err = reached - target

    print(f"\n  reached {reached:+.3f} rad   (error {err:+.3f})")
    if abs(err) > 0.08:
        print("  LARGE TRACKING ERROR -- the joint could not get there.")
        print("  That means a mechanical limit, a bind, or not enough torque.")
        print("  It does NOT mean a wrong sign; signs cancel in software.")
    else:
        print("  Tracking good.")

    print("\n" + "=" * 68)
    print("  NOW LOOK AT THE ROBOT.")
    print("=" * 68)
    print(f"""
  You commanded {joint} by {delta:+.3f} rad.

  Did it move the direction a POSITIVE angle should move in the URDF?

    * yes  -> sign {sign:+d} is correct for this joint
    * no   -> flip this joint in SERVO_SIGNS in robot_config.py

  If you are unsure which way positive is for this joint, open the URDF
  and read its axis, or compare against the sim.
""")
    input("Press Enter to return to start and release torque... ")

    try:
        for i in range(steps, -1, -1):
            goal = list(pos)
            goal[idx] = start + delta * (i / steps)
            bus.write_targets(goal)
            time.sleep(0.03)
    finally:
        torque_one(bus, dxl_id, False)
        bus.close()
        print("Returned and torque off.")


def cmd_pose(bus, which):
    if which == "crouch":
        target = np.array([cfg.DEFAULT_JOINT_POS[n] for n in cfg.JOINT_NAMES],
                          dtype=np.float32)
        look = """
  The trained crouch should look like:
    * both knees bent FORWARD, the same amount as each other
    * both feet FLAT, soles parallel to the ground
    * hips, leg yaws and ankle rolls all neutral -- no splay, no twist
    * left and right MIRRORED, visibly symmetric

  If one knee bends backward while the other bends forward, that joint's
  sign is wrong. Symmetry is the tell.
"""
    else:
        target = np.zeros(14, dtype=np.float32)
        look = """
  URDF zero should look like:
    * legs completely STRAIGHT
    * feet FLAT, perpendicular to the shins
    * legs PARALLEL, no splay or twist

  This should match the pose you held during calibrate.py zero. If it
  doesn't, the offsets are off.
"""

    pos, _ = bus.read_state()
    pos = np.array(pos, dtype=np.float32)
    biggest = float(np.abs(target - pos).max())

    print(f"\n  Pose: {which}")
    print(f"  Largest single-joint move: {biggest:.3f} rad")
    print(f"  Ramp time: {cfg.RAMP_SECONDS:.1f} s")
    print("\n  ALL 14 servos will be energized.")

    if not confirm("Robot ON A STAND, feet clear, hand on the power switch?"):
        bus.close()
        sys.exit("Aborted. No torque was enabled.")

    bus.configure_position_mode()         # leaves torque off
    bus.write_targets(pos)                # goal = here, no jump
    bus.set_torque(True)
    print("\n  torque on, holding present position.")
    time.sleep(0.5)

    steps = max(1, int(cfg.RAMP_SECONDS * cfg.CONTROL_HZ))
    dt = 1.0 / cfg.CONTROL_HZ
    print(f"  ramping over {cfg.RAMP_SECONDS:.1f}s -- Ctrl+C aborts\n")
    try:
        for i in range(1, steps + 1):
            a = i / steps
            bus.write_targets((1.0 - a) * pos + a * target)
            if i % int(cfg.CONTROL_HZ / 2) == 0:
                print(f"    {100 * a:5.1f}%")
            time.sleep(dt)
    except KeyboardInterrupt:
        bus.set_torque(False)
        bus.close()
        sys.exit("\nAborted. Torque off.")

    time.sleep(SETTLE_S)
    pos2, _ = bus.read_state()
    pos2 = np.array(pos2, dtype=np.float32)
    err = pos2 - target

    print("\n  joint                 target   reached     error")
    print("  " + "-" * 50)
    for i, n in enumerate(cfg.JOINT_NAMES):
        flag = "  <-- can't reach" if abs(err[i]) > 0.08 else ""
        print(f"  {n:<20} {target[i]:+7.3f}  {pos2[i]:+8.3f}  "
              f"{err[i]:+8.3f}{flag}")

    worst = int(np.argmax(np.abs(err)))
    print(f"\n  worst error: {cfg.JOINT_NAMES[worst]} {err[worst]:+.3f} rad")
    print(look)
    print("=" * 68)
    try:
        input("Holding. Press Enter to release torque (or Ctrl+C)... ")
    except KeyboardInterrupt:
        pass
    finally:
        bus.set_torque(False)
        bus.close()
        print("Torque off -- joints are limp. SUPPORT THE ROBOT.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["one", "crouch", "zero"])
    ap.add_argument("joint", nargs="?", default=None)
    ap.add_argument("delta", nargs="?", type=float, default=0.25)
    args = ap.parse_args()

    print("=" * 68)
    print("  hold_pose.py -- TORQUE WILL BE ENABLED")
    print("=" * 68)

    bus = open_bus()

    if args.mode == "one":
        if not args.joint:
            bus.close()
            sys.exit("one <joint> [delta]  -- joint name required")
        cmd_one(bus, args.joint, args.delta)
    else:
        cmd_pose(bus, args.mode)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
