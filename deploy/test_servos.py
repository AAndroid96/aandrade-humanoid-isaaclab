#!/usr/bin/env python3
"""test_servos.py -- can the Jetson see the 14 XL330s?

Self-contained: no robot_config, no local imports. Only needs the official
ROBOTIS SDK.  pip3 install dynamixel-sdk

Assumes the OpenRB-150 is running ROBOTIS's STOCK usb_to_dynamixel
pass-through sketch, so the OpenRB is a transparent wire and all Protocol
2.0 work happens here.

SAFETY: never enables torque, never writes a goal position. Read-only.
Nothing can move as a result of running this.

Usage:
    python3 test_servos.py                       # /dev/ttyACM1 @ 1 Mbps
    python3 test_servos.py /dev/ttyACM0
    python3 test_servos.py /dev/ttyUSB0 57600    # U2D2, or fallback baud
"""

import sys

try:
    from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler
except ImportError:
    sys.exit("dynamixel-sdk missing.  pip3 install dynamixel-sdk")

# --- XL330-M288-T control table (Protocol 2.0) ---------------------------
ADDR_PRESENT_POSITION = 132
ADDR_TORQUE_ENABLE = 64

TICKS_PER_REV = 4096
TWO_PI = 6.283185307179586
TICKS_PER_RAD = TICKS_PER_REV / TWO_PI

# Policy joint order. PhysX orders joints breadth-first, which is why left
# and right interleave. Verified 2026-07-26 against the trained env.
# (name, dxl_id, sign)  -- sign: +1 left, -1 right (mirrored assembly)
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

OFFSET = 2048          # tick reading at URDF zero (centered horn)


def ticks_to_rad(ticks, sign, offset=OFFSET):
    return sign * (ticks - offset) / TICKS_PER_RAD


def main():
    port_name = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM1"
    baud = int(sys.argv[2]) if len(sys.argv) > 2 else 1000000

    print(f"Opening {port_name} at {baud} baud...\n")

    port = PortHandler(port_name)
    if not port.openPort():
        sys.exit(
            f"Could not open {port_name}.\n"
            "  ls /dev/ttyACM* /dev/ttyUSB*   to see what exists\n"
            "  Is the OpenRB running usb_to_dynamixel?"
        )
    if not port.setBaudRate(baud):
        port.closePort()
        sys.exit(f"Could not set baud {baud}.")

    packet = PacketHandler(2.0)

    found, missing = [], []
    print(f"  {'joint':<20} {'id':>3}  {'result':<8} {'position':>12}")
    print(f"  {'-'*20} {'-'*3}  {'-'*8} {'-'*12}")

    for name, dxl_id, sign in JOINTS:
        _, res, err = packet.ping(port, dxl_id)
        if res != COMM_SUCCESS or err != 0:
            missing.append((name, dxl_id))
            print(f"  {name:<20} {dxl_id:>3}  {'MISS':<8} {'-':>12}")
            continue

        found.append((name, dxl_id))
        raw, res2, err2 = packet.read4ByteTxRx(
            port, dxl_id, ADDR_PRESENT_POSITION)
        if res2 != COMM_SUCCESS or err2 != 0:
            print(f"  {name:<20} {dxl_id:>3}  {'OK':<8} {'read err':>12}")
            continue

        rad = ticks_to_rad(raw, sign)
        print(f"  {name:<20} {dxl_id:>3}  {'OK':<8} "
              f"{raw:>5} tk {rad:+6.3f} rad")

    print(f"\n{len(found)} of {len(JOINTS)} servos answered.")

    if not found:
        print(
            "\nNothing answered. Most likely causes, in order:\n"
            "  1. NO SERVO POWER. 14 XL330s cannot run from USB. Feed the\n"
            "     OpenRB terminal block 5 V (NEVER 12 V) and set the power\n"
            "     jumper to VIN(DXL) -- external power does not reach the\n"
            "     bus otherwise.\n"
            "  2. The OpenRB is not running the stock usb_to_dynamixel.\n"
            "  3. Baud mismatch. Servos were set to 1 Mbps in July; retry\n"
            "     with: python3 test_servos.py <port> 57600\n"
        )
    elif missing:
        names = ", ".join(f"{n}(id {i})" for n, i in missing)
        print(f"Missing: {names}")
        print("\nSome answered, so the LINK AND POWER ARE GOOD. Missing "
              "servos are a\nwiring break in that chain, or a wrong ID on "
              "those units.")
    else:
        print("\nAll 14 present. Torque was never enabled -- nothing moved.")
        print("Move a joint by hand and rerun; the tick values should "
              "track.\nThat confirms real end-to-end readback.")

    # Report torque state without changing it, so a surprise is visible.
    if found:
        on = []
        for name, dxl_id in found:
            val, res, err = packet.read1ByteTxRx(
                port, dxl_id, ADDR_TORQUE_ENABLE)
            if res == COMM_SUCCESS and err == 0 and val == 1:
                on.append(name)
        if on:
            print(f"\nNOTE: torque is already ON for: {', '.join(on)}")
            print("This script did not enable it. Those joints are holding "
                  "position.")

    port.closePort()


if __name__ == "__main__":
    main()
