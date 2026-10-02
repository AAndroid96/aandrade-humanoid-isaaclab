#!/usr/bin/env python3
"""calibrate.py -- measure real servo offsets and verify sign conventions.

READ-ONLY. Never enables torque, never writes a goal position. Nothing can
move because of this script; you move the robot by hand.

Two jobs:

  zero   Capture the tick reading of every joint while the robot is held in
         the URDF reference pose, and emit a SERVO_OFFSETS dict to paste
         into robot_config.py. Replaces the 2048 guess with measurement.

  watch  Stream one joint (or all) live, so you can move it by hand and see
         which way the reported angle goes. This is how you verify the +1/-1
         signs WITHOUT enabling torque.

Usage:
    python3 calibrate.py zero                     # auto-detect port
    python3 calibrate.py zero /dev/ttyACM0
    python3 calibrate.py watch left_knee1_pitch
    python3 calibrate.py watch all
"""

import glob
import sys
import time

try:
    from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler
except ImportError:
    sys.exit("dynamixel-sdk missing.  pip3 install dynamixel-sdk")

ADDR_PRESENT_POSITION = 132
TICKS_PER_REV = 4096
TWO_PI = 6.283185307179586
TICKS_PER_RAD = TICKS_PER_REV / TWO_PI

NOMINAL_OFFSET = 2048
SAMPLES = 12                  # averaged per joint, to beat sensor noise
SPREAD_WARN = 8               # ticks of spread that means "it's still moving"

# (name, dxl_id, sign) in policy joint order.
JOINTS = [
    ("left_hip_pitch",     1, +1),
    ("right_hip_pitch",   11, -1),
    ("left_hip_roll",      2, +1),
    ("right_hip_roll",    12, -1),
    ("left_leg_yaw",       3, +1),
    ("right_leg_yaw",     13, -1),
    ("left_knee1_pitch",   4, +1),
    ("right_knee1_pitch", 14, -1),
    ("left_knee2_pitch",   5, +1),
    ("right_knee2_pitch", 15, -1),
    ("left_ankle_pitch",   6, +1),
    ("right_ankle_pitch", 16, -1),
    ("left_ankle_roll",    7, +1),
    ("right_ankle_roll",  17, -1),
]

BY_NAME = {n: (i, s) for n, i, s in JOINTS}


def autoport():
    for pattern in ("/dev/ttyACM*", "/dev/ttyUSB*"):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    sys.exit("No /dev/ttyACM* or /dev/ttyUSB* found. Is the OpenRB plugged in?")


def open_bus(port_name, baud=1000000):
    port = PortHandler(port_name)
    if not port.openPort():
        sys.exit(f"Could not open {port_name}")
    if not port.setBaudRate(baud):
        port.closePort()
        sys.exit(f"Could not set baud {baud}")
    return port, PacketHandler(2.0)


def read_ticks(packet, port, dxl_id):
    raw, res, err = packet.read4ByteTxRx(port, dxl_id, ADDR_PRESENT_POSITION)
    if res != COMM_SUCCESS or err != 0:
        return None
    return raw


def sample(packet, port, dxl_id, n=SAMPLES):
    """Return (mean_ticks, spread) or (None, None) on failure."""
    vals = []
    for _ in range(n):
        t = read_ticks(packet, port, dxl_id)
        if t is not None:
            vals.append(t)
        time.sleep(0.004)
    if not vals:
        return None, None
    return sum(vals) / len(vals), max(vals) - min(vals)


def cmd_zero(port_name):
    print(__doc__.split("Usage:")[0].strip())
    print("\n" + "=" * 68)
    print("REFERENCE POSE -- hold the robot like this before continuing:")
    print("=" * 68)
    print("""
  * Legs STRAIGHT   -- hips, both knee joints and ankles all unbent
  * Feet FLAT       -- sole perpendicular to the shin, as if standing
  * Legs PARALLEL   -- no splay at the hip rolls, no twist at the leg yaws
  * Robot SUPPORTED -- on a stand or held; it must not be bearing weight
                       unevenly or sagging under gravity

This pose IS "all joints = 0 radians" in the URDF. Whatever ticks the
servos report while you hold it become the offsets.

Torque is OFF, so the joints are limp and you can pose them freely.
""")
    input("Hold the pose, then press Enter (Ctrl+C to abort)... ")

    port, packet = open_bus(port_name)
    print(f"\nSampling {SAMPLES}x per joint...\n")
    print(f"  {'joint':<20} {'ticks':>7} {'vs 2048':>9} {'spread':>7}")
    print(f"  {'-'*20} {'-'*7} {'-'*9} {'-'*7}")

    offsets, noisy, failed = {}, [], []

    for name, dxl_id, _sign in JOINTS:
        mean, spread = sample(packet, port, dxl_id)
        if mean is None:
            failed.append(name)
            print(f"  {name:<20} {'FAILED':>7}")
            continue
        ticks = int(round(mean))
        offsets[name] = ticks
        delta = ticks - NOMINAL_OFFSET
        flag = ""
        if spread > SPREAD_WARN:
            noisy.append(name)
            flag = "  <-- moving?"
        print(f"  {name:<20} {ticks:>7} {delta:>+9} {spread:>7}{flag}")

    port.closePort()

    if failed:
        print(f"\nFAILED to read: {', '.join(failed)}")
        print("Fix those before trusting this calibration.")
        return

    if noisy:
        print(f"\nWARNING: unstable readings on {', '.join(noisy)}")
        print(f"Spread above {SPREAD_WARN} ticks means the joint was still "
              "settling.\nRe-run while holding it steadier.")

    big = [(n, v - NOMINAL_OFFSET) for n, v in offsets.items()
           if abs(v - NOMINAL_OFFSET) > 200]
    if big:
        print("\nNOTE: these sit far from a centered horn:")
        for n, d in big:
            print(f"  {n:<20} {d:+5} ticks ({d / TICKS_PER_RAD:+.3f} rad)")
        print("Either the horn was installed off-center (fine -- that is "
              "exactly what\nthis calibration captures), or the pose was not "
              "actually at zero.\nIf a value surprises you, check that joint "
              "physically before accepting.")

    print("\n" + "=" * 68)
    print("Paste this over SERVO_OFFSETS in robot_config.py:")
    print("=" * 68 + "\n")
    print("SERVO_OFFSETS = {")
    width = max(len(n) for n in offsets) + 3
    for name, _id, _s in JOINTS:
        key = f'"{name}":'
        print(f"    {key:<{width}} {offsets[name]},")
    print("}")


def cmd_watch(port_name, which):
    port, packet = open_bus(port_name)

    if which == "all":
        watching = JOINTS
    elif which in BY_NAME:
        dxl_id, sign = BY_NAME[which]
        watching = [(which, dxl_id, sign)]
    else:
        port.closePort()
        print(f"Unknown joint '{which}'. Valid names:")
        for n, _, _ in JOINTS:
            print(f"  {n}")
        return

    print("\nSIGN CHECK -- torque is OFF, move the joint BY HAND.\n")
    print("Decide what POSITIVE means for each joint from the URDF, then")
    print("move it that way and watch the sign. If it goes the wrong way,")
    print("flip that joint's entry in SERVO_SIGNS.\n")
    print("Ctrl+C to stop.\n")

    try:
        while True:
            parts = []
            for name, dxl_id, sign in watching:
                t = read_ticks(packet, port, dxl_id)
                if t is None:
                    parts.append(f"{name}=ERR")
                    continue
                rad = sign * (t - NOMINAL_OFFSET) / TICKS_PER_RAD
                parts.append(f"{name}={t:>4}tk {rad:+.3f}rad")
            if len(parts) == 1:
                print("\r  " + parts[0] + "        ", end="", flush=True)
            else:
                print("\n".join("  " + p for p in parts))
                print("-" * 40)
            time.sleep(0.15 if len(parts) == 1 else 0.5)
    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        port.closePort()


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("zero", "watch"):
        sys.exit(__doc__)

    mode = sys.argv[1]

    if mode == "zero":
        port_name = sys.argv[2] if len(sys.argv) > 2 else autoport()
        cmd_zero(port_name)
    else:
        if len(sys.argv) < 3:
            sys.exit("watch needs a joint name, or 'all'")
        which = sys.argv[2]
        port_name = sys.argv[3] if len(sys.argv) > 3 else autoport()
        cmd_watch(port_name, which)


if __name__ == "__main__":
    main()
