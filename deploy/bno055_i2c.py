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

    def __init__(self, bus: int = 7, address: int = 0x28, retries: int = 4):
        self.address = address
        self._bus_num = bus
        self._retries = retries
        # I2C RESILIENCE counters -- read these to judge bus health.
        self.io_errors = 0        # transactions that raised
        self.reopens = 0          # times the bus was closed and reopened
        self.bus = SMBus(bus)
        self._init_sensor()

    def _reopen(self) -> None:
        """Close and reopen the I2C bus. Clears a wedged controller after a
        noise-induced error, which a bare retry often cannot."""
        self.reopens += 1
        try:
            self.bus.close()
        except Exception:
            pass
        time.sleep(0.005)
        self.bus = SMBus(self._bus_num)

    def _io(self, fn):
        """Run one I2C transaction with retries and a bus reopen.

        Raises the last OSError once every attempt is spent, so the caller's
        safety watchdog still fires on a genuinely dead bus."""
        last = None
        for attempt in range(self._retries):
            try:
                return fn()
            except OSError as exc:
                last = exc
                self.io_errors += 1
                if attempt == self._retries - 2:
                    self._reopen()
                else:
                    time.sleep(0.002)
        raise last

    # -- setup --------------------------------------------------------------
    def _write(self, reg: int, value: int) -> None:
        self._io(lambda: self.bus.write_byte_data(self.address, reg, value))

    def _read(self, reg: int, length: int = 1):
        if length == 1:
            return self._io(lambda: self.bus.read_byte_data(self.address, reg))
        return self._io(
            lambda: self.bus.read_i2c_block_data(self.address, reg, length))

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
