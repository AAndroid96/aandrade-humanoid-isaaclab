#!/usr/bin/env python3
"""run_policy.py — Jetson-side brain for AAndrade-Humanoid-Lite.

50 Hz loop:  read state packet from OpenRB-150  ->  build observation vector
             ->  ONNX policy inference  ->  send 14 joint targets back down.

Keyboard teleop (terminal must have focus):
    w/s : forward / backward     (forward = -Y on this robot)
    a/d : strafe left / right
    q/e : turn left / right
  space : zero all commands
      k : KILL — torque off all servos immediately
    ESC : kill + exit

SAFETY: start with the robot ON A STAND, feet off the ground.
"""

import os
import struct
import sys
import termios
import time
import tty
import select

import numpy as np
import onnxruntime as ort
import serial

import robot_config as cfg

NUM_JOINTS = 14
STATE_LEN = 74
CMD_LEN = 32

# ---------------------------------------------------------------------------
# Serial protocol
# ---------------------------------------------------------------------------

def xor_checksum(data: bytes) -> int:
    x = 0
    for b in data:
        x ^= b
    return x


def send_targets(port: serial.Serial, targets_rad: np.ndarray) -> None:
    mrad = np.clip(targets_rad * 1000.0, -32000, 32000).astype("<i2")
    body = bytes([0x01]) + mrad.tobytes()
    port.write(b"\xaa\x55" + body + bytes([xor_checksum(body)]))


def send_kill(port: serial.Serial) -> None:
    body = bytes([0x02])
    port.write(b"\xaa\x55" + body + bytes([xor_checksum(body)]))


class StateParser:
    """Incremental parser for 74-byte STATE packets."""

    def __init__(self):
        self.buf = bytearray()
        self.latest = None  # (joint_pos[14], joint_vel[14], quat[4], gyro[3])

    def feed(self, data: bytes) -> None:
        self.buf.extend(data)
        while True:
            idx = self.buf.find(b"\xaa\x55")
            if idx < 0:
                del self.buf[:-1]
                return
            if idx > 0:
                del self.buf[:idx]
            if len(self.buf) < STATE_LEN:
                return
            pkt = bytes(self.buf[:STATE_LEN])
            if pkt[2] == 0x10 and xor_checksum(pkt[2:73]) == pkt[73]:
                vals = struct.unpack("<35h", pkt[3:73])
                self.latest = (
                    np.array(vals[0:14], dtype=np.float32) / 1000.0,     # rad
                    np.array(vals[14:28], dtype=np.float32) / 1000.0,    # rad/s
                    np.array(vals[28:32], dtype=np.float32) / 10000.0,   # quat wxyz
                    np.array(vals[32:35], dtype=np.float32) / 1000.0,    # rad/s
                )
                del self.buf[:STATE_LEN]
            else:
                del self.buf[:2]  # bad packet: resync past this header


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def projected_gravity(quat_wxyz: np.ndarray, imu_to_base: np.ndarray) -> np.ndarray:
    """World gravity direction (0,0,-1) expressed in the base frame."""
    w, x, y, z = quat_wxyz
    # Rotation matrix of the IMU frame in the world, then g_body = R^T @ g_world.
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)
    g_imu = R.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return imu_to_base @ g_imu


# ---------------------------------------------------------------------------
# Keyboard (raw terminal, non-blocking)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    default_pos = np.array(
        [cfg.DEFAULT_JOINT_POS[n] for n in cfg.JOINT_NAMES], dtype=np.float32
    )
    imu_to_base = np.array(cfg.IMU_TO_BASE, dtype=np.float32)

    sess = ort.InferenceSession(os.path.expanduser(cfg.POLICY_PATH))
    in_name = sess.get_inputs()[0].name
    obs_dim = sess.get_inputs()[0].shape[-1]
    expected = 54 if cfg.INCLUDE_BASE_LIN_VEL else 51
    if obs_dim != expected:
        sys.exit(
            f"Policy expects obs dim {obs_dim} but config builds {expected}. "
            f"Check INCLUDE_BASE_LIN_VEL in robot_config.py."
        )

    port = serial.Serial(cfg.SERIAL_PORT, 115200, timeout=0)
    parser = StateParser()
    prev_action = np.zeros(NUM_JOINTS, dtype=np.float32)
    command = np.zeros(3, dtype=np.float32)  # vx, vy, wz
    dt = 1.0 / cfg.CONTROL_HZ

    print(f"Policy loaded ({obs_dim} obs). Waiting for state from OpenRB...")
    while parser.latest is None:
        parser.feed(port.read(256))
        time.sleep(0.01)
    print("State stream up. w/s a/d q/e drive, space=stop, k=KILL, ESC=exit.")

    with Keyboard() as kb:
        next_tick = time.monotonic()
        while True:
            # --- keyboard ---
            for key in kb.poll():
                if key == "w":   command[1] = -cfg.FWD_SPEED   # forward = -Y
                elif key == "s": command[1] = +cfg.FWD_SPEED
                elif key == "a": command[0] = +cfg.LAT_SPEED
                elif key == "d": command[0] = -cfg.LAT_SPEED
                elif key == "q": command[2] = +cfg.YAW_SPEED
                elif key == "e": command[2] = -cfg.YAW_SPEED
                elif key == " ": command[:] = 0.0
                elif key == "k":
                    send_kill(port)
                    print("\nKILL sent — servos limp. ESC to exit, any drive key to re-arm.")
                elif key == "\x1b":  # ESC
                    send_kill(port)
                    print("\nKill + exit.")
                    return

            # --- state ---
            parser.feed(port.read(1024))
            joint_pos, joint_vel, quat, gyro = parser.latest
            base_ang_vel = imu_to_base @ gyro
            grav = projected_gravity(quat, imu_to_base)

            # --- observation (order must match training) ---
            parts = []
            if cfg.INCLUDE_BASE_LIN_VEL:
                parts.append(np.zeros(3, dtype=np.float32))  # unmeasurable: zeros
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
            targets = default_pos + cfg.ACTION_SCALE * action
            send_targets(port, targets)

            # --- 50 Hz pacing ---
            next_tick += dt
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()  # fell behind: resync, don't spiral


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Ctrl+C: watchdog on the OpenRB torques off within 200 ms anyway,
        # but try to be explicit.
        try:
            serial.Serial(cfg.SERIAL_PORT, 115200, timeout=0.1).write(
                b"\xaa\x55\x02" + bytes([0x02])
            )
        except Exception:
            pass
        print("\nInterrupted — kill sent.")
