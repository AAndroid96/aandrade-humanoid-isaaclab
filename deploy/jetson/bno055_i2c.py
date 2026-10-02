"""Minimal BNO055 driver over raw I2C (smbus2) for the Jetson Orin Nano.

Deliberately dependency-light: no Blinka, no CircuitPython stack. Just the
handful of registers the locomotion policy needs — orientation quaternion
and body-frame angular velocity.

Wiring (Jetson 40-pin header):
    BNO055 VIN -> pin 1  (3.3V)
    BNO055 GND -> pin 6  (GND)
    BNO055 SDA -> pin 3  (I2C bus 7 on Orin Nano)
    BNO055 SCL -> pin 5

No other BNO055 pin connects to anything.
"""

from __future__ import annotations

import struct
import time

from smbus2 import SMBus

# --- Register map (page 0) -------------------------------------------------
REG_CHIP_ID = 0x00
REG_PAGE_ID = 0x07
REG_GYRO_DATA = 0x14      # 6 bytes: x, y, z  int16 LE
REG_QUAT_DATA = 0x20      # 8 bytes: w, x, y, z int16 LE
REG_CALIB_STAT = 0x35
REG_UNIT_SEL = 0x3B
REG_OPR_MODE = 0x3D
REG_PWR_MODE = 0x3E
REG_SYS_TRIGGER = 0x3F

CHIP_ID_EXPECTED = 0xA0

MODE_CONFIG = 0x00
MODE_NDOF = 0x0C
MODE_IMUPLUS = 0x08       # gyro+accel only: no magnetometer, no mag calibration
PWR_NORMAL = 0x00

QUAT_SCALE = 1.0 / (1 << 14)      # 2^14 LSB per unit quaternion
GYRO_DPS_PER_LSB = 1.0 / 16.0     # default UNIT_SEL: 16 LSB per deg/s
DEG_TO_RAD = 3.14159265358979 / 180.0


class BNO055Error(RuntimeError):
    pass


class BNO055:
    """Blocking, non-hanging BNO055 reader.

    Every call goes through smbus2, which raises OSError on a bus problem
    rather than spinning forever the way the Arduino Wire library does.
    """

    def __init__(self, bus: int = 7, address: int = 0x28):
        self.address = address
        self.bus = SMBus(bus)
        self._init_sensor()

    # -- setup --------------------------------------------------------------
    def _write(self, reg: int, value: int) -> None:
        self.bus.write_byte_data(self.address, reg, value)

    def _read(self, reg: int, length: int = 1):
        if length == 1:
            return self.bus.read_byte_data(self.address, reg)
        return self.bus.read_i2c_block_data(self.address, reg, length)

    def _init_sensor(self) -> None:
        chip_id = self._read(REG_CHIP_ID)
        if chip_id != CHIP_ID_EXPECTED:
            raise BNO055Error(
                f"chip id 0x{chip_id:02X}, expected 0x{CHIP_ID_EXPECTED:02X} "
                "— wrong device or bad wiring"
            )

        self._write(REG_PAGE_ID, 0x00)
        self._write(REG_OPR_MODE, MODE_CONFIG)
        time.sleep(0.03)                      # config switch takes ~19 ms
        self._write(REG_PWR_MODE, PWR_NORMAL)
        time.sleep(0.01)
        self._write(REG_UNIT_SEL, 0x00)       # deg/s, m/s^2, degrees, Celsius
        self._write(REG_SYS_TRIGGER, 0x00)    # internal crystal
        time.sleep(0.01)
        self._write(REG_OPR_MODE, MODE_IMUPLUS)  # relative heading, mag ignored
        time.sleep(0.03)                      # mode switch takes ~7 ms

    # -- reads --------------------------------------------------------------
    def read_quaternion(self):
        """Orientation as (w, x, y, z), normalized."""
        raw = self._read(REG_QUAT_DATA, 8)
        w, x, y, z = struct.unpack("<hhhh", bytes(raw))
        q = (w * QUAT_SCALE, x * QUAT_SCALE, y * QUAT_SCALE, z * QUAT_SCALE)
        norm = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2) ** 0.5
        if norm < 1e-6:
            # Sensor not yet producing a fix; identity keeps the math sane.
            return (1.0, 0.0, 0.0, 0.0)
        return (q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm)

    def read_gyro(self):
        """Body-frame angular velocity as (x, y, z) in rad/s."""
        raw = self._read(REG_GYRO_DATA, 6)
        x, y, z = struct.unpack("<hhh", bytes(raw))
        s = GYRO_DPS_PER_LSB * DEG_TO_RAD
        return (x * s, y * s, z * s)

    def read_state(self):
        """One shot: (quat_wxyz, gyro_xyz_rad_s)."""
        return self.read_quaternion(), self.read_gyro()

    def calibration(self):
        """(sys, gyro, accel, mag), each 0-3. In IMUPLUS mag stays 0 by design."""
        stat = self._read(REG_CALIB_STAT)
        return ((stat >> 6) & 0x03, (stat >> 4) & 0x03,
                (stat >> 2) & 0x03, stat & 0x03)

    def close(self) -> None:
        try:
            self.bus.close()
        except Exception:
            pass
