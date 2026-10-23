#!/usr/bin/env python3
"""test_imu.py — verify the BNO055 on the Jetson, and check its mounting.

No motors, no serial, no policy. Just reads the sensor and prints what the
locomotion policy would see, so you can confirm the axes before trusting it.

Run:  python3 test_imu.py        (Ctrl+C to stop)

WHAT TO CHECK
  1. Robot upright and still -> gravity reads about ( 0.00,  0.00, -1.00)
  2. Tip the robot NOSE DOWN  -> the Y component goes clearly NEGATIVE
     (this robot's forward is -Y; established during training)
  3. Tip it onto its left side -> the X component moves clearly off zero
  4. Rotate it ~90 deg in about a second -> peak gyro near 1.5 rad/s
     (if you see ~90 instead, the units are wrong — tell me)

If gravity does not sit near (0, 0, -1) when upright, the IMU is mounted
rotated relative to the robot body. That is fixable in one line
(IMU_TO_BASE in robot_config.py) — just report what you saw.
"""

import sys
import time

import numpy as np

import robot_config as cfg
from bno055_i2c import BNO055, BNO055Error


def projected_gravity(quat_wxyz, imu_to_base):
    """World gravity (0,0,-1) expressed in the robot's base frame."""
    w, x, y, z = quat_wxyz
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)
    g_imu = R.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return imu_to_base @ g_imu


def main():
    imu_to_base = np.array(cfg.IMU_TO_BASE, dtype=np.float32)

    try:
        imu = BNO055(bus=cfg.IMU_I2C_BUS, address=cfg.IMU_I2C_ADDR)
    except (BNO055Error, OSError) as exc:
        sys.exit(
            f"Could not open the BNO055 on bus {cfg.IMU_I2C_BUS} "
            f"at 0x{cfg.IMU_I2C_ADDR:02X}: {exc}\n"
            f"Check wiring, and that 'i2cdetect -y -r {cfg.IMU_I2C_BUS}' "
            "still shows 28."
        )

    print("BNO055 online. Ctrl+C to stop.\n")
    print(f"{'gravity x':>10} {'y':>8} {'z':>8}   "
          f"{'gyro x':>8} {'y':>8} {'z':>8}   calib(s,g,a,m)")

    peak_gyro = 0.0
    try:
        while True:
            quat, gyro = imu.read_state()
            grav = projected_gravity(quat, imu_to_base)
            w = imu_to_base @ np.array(gyro, dtype=np.float32)
            peak_gyro = max(peak_gyro, float(np.abs(w).max()))
            c = imu.calibration()
            print(f"\r{grav[0]:10.2f} {grav[1]:8.2f} {grav[2]:8.2f}   "
                  f"{w[0]:8.2f} {w[1]:8.2f} {w[2]:8.2f}   "
                  f"{c[0]}{c[1]}{c[2]}{c[3]}  peak|w|={peak_gyro:.2f}",
                  end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print(f"\n\nPeak angular rate seen: {peak_gyro:.2f} rad/s")
        imu.close()


if __name__ == "__main__":
    main()
