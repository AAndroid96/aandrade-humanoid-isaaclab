"""Dynamixel bus access for AAndrade-Humanoid-Lite.

Talks Protocol 2.0 directly from the Jetson using ROBOTIS's official SDK,
through an OpenRB-150 running the stock `usb_to_dynamixel` pass-through
sketch. The OpenRB is a transparent wire — no custom firmware involved.

Install:  pip install dynamixel-sdk
"""

from __future__ import annotations

import struct

from dynamixel_sdk import (
    COMM_SUCCESS,
    GroupSyncRead,
    GroupSyncWrite,
    PacketHandler,
    PortHandler,
)

# --- XL330-M288-T control table (Protocol 2.0) -----------------------------
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132

LEN_GOAL_POSITION = 4
LEN_PRESENT_BLOCK = 8          # velocity (128,4) + position (132,4), adjacent

MODE_POSITION = 3

TICKS_PER_REV = 4096
VEL_UNIT_RPM = 0.229           # one velocity LSB = 0.229 rpm

TWO_PI = 6.283185307179586


def _s32(v: int) -> int:
    """Interpret a 32-bit unsigned word as signed."""
    return v - (1 << 32) if v >= (1 << 31) else v


class DynamixelBus:
    """Sync-read joint state and sync-write joint targets for N servos.

    Angles are in URDF joint space (radians), converted per joint using the
    calibration table: ticks = offset + sign * radians * TICKS_PER_RAD.
    """

    def __init__(self, port_name: str, baudrate: int, calib):
        """calib: list of (dxl_id, sign, offset), in policy joint order."""
        self.calib = list(calib)
        self.ids = [c[0] for c in self.calib]

        self.port = PortHandler(port_name)
        if not self.port.openPort():
            raise RuntimeError(f"could not open {port_name}")
        if not self.port.setBaudRate(baudrate):
            raise RuntimeError(f"could not set baudrate {baudrate}")

        self.packet = PacketHandler(2.0)
        self.ticks_per_rad = TICKS_PER_REV / TWO_PI

        self.sync_write = GroupSyncWrite(
            self.port, self.packet, ADDR_GOAL_POSITION, LEN_GOAL_POSITION
        )
        self.sync_read = GroupSyncRead(
            self.port, self.packet, ADDR_PRESENT_VELOCITY, LEN_PRESENT_BLOCK
        )
        for dxl_id in self.ids:
            if not self.sync_read.addParam(dxl_id):
                raise RuntimeError(f"sync read addParam failed for id {dxl_id}")

        # Last good readings, so one dropped packet doesn't blank the state.
        self._pos_rad = [0.0] * len(self.ids)
        self._vel_rad = [0.0] * len(self.ids)

    # -- discovery ----------------------------------------------------------
    def ping_all(self):
        """Return the list of servo IDs that answered a ping."""
        found = []
        for dxl_id in self.ids:
            _, result, err = self.packet.ping(self.port, dxl_id)
            if result == COMM_SUCCESS and err == 0:
                found.append(dxl_id)
        return found

    # -- configuration ------------------------------------------------------
    def set_torque(self, on: bool) -> None:
        for dxl_id in self.ids:
            self.packet.write1ByteTxRx(
                self.port, dxl_id, ADDR_TORQUE_ENABLE, 1 if on else 0
            )

    def configure_position_mode(self) -> None:
        """Torque must be OFF to change operating mode."""
        self.set_torque(False)
        for dxl_id in self.ids:
            self.packet.write1ByteTxRx(
                self.port, dxl_id, ADDR_OPERATING_MODE, MODE_POSITION
            )

    # -- conversions --------------------------------------------------------
    def _rad_to_ticks(self, i: int, rad: float) -> int:
        _, sign, offset = self.calib[i]
        # round, not truncate: truncation biases toward zero and doubles the
        # worst-case quantization error (one tick = 2*pi/4096 = 0.0015 rad).
        return int(round(offset + sign * rad * self.ticks_per_rad))

    def _ticks_to_rad(self, i: int, ticks: int) -> float:
        _, sign, offset = self.calib[i]
        return sign * (ticks - offset) / self.ticks_per_rad

    # -- I/O ----------------------------------------------------------------
    def write_targets(self, targets_rad) -> bool:
        self.sync_write.clearParam()
        for i, dxl_id in enumerate(self.ids):
            ticks = self._rad_to_ticks(i, float(targets_rad[i]))
            ticks = max(0, min(TICKS_PER_REV - 1, ticks))   # never wrap around
            self.sync_write.addParam(dxl_id, struct.pack("<i", ticks))
        return self.sync_write.txPacket() == COMM_SUCCESS

    def read_state(self):
        """Return (positions_rad, velocities_rad_s). Stale values on dropout."""
        if self.sync_read.txRxPacket() != COMM_SUCCESS:
            return self._pos_rad, self._vel_rad

        vel_scale = VEL_UNIT_RPM * TWO_PI / 60.0     # rpm -> rad/s
        for i, dxl_id in enumerate(self.ids):
            if not self.sync_read.isAvailable(
                dxl_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_BLOCK
            ):
                continue
            raw_vel = self.sync_read.getData(dxl_id, ADDR_PRESENT_VELOCITY, 4)
            raw_pos = self.sync_read.getData(dxl_id, ADDR_PRESENT_POSITION, 4)
            _, sign, _ = self.calib[i]
            self._vel_rad[i] = sign * _s32(raw_vel) * vel_scale
            self._pos_rad[i] = self._ticks_to_rad(i, raw_pos)
        return self._pos_rad, self._vel_rad

    def close(self) -> None:
        try:
            self.set_torque(False)
        except Exception:
            pass
        try:
            self.port.closePort()
        except Exception:
            pass
