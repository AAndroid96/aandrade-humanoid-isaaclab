# AAndrade-Humanoid-Lite — Deployment Package

Jetson Orin Nano (brain) + OpenRB-150 (spinal cord), connected over USB serial.

```
[gamepad/keyboard] -> Jetson: run_policy.py (ONNX @ 50 Hz)
                        |  USB serial: 14 joint targets (CMD, 32 B)
                        v
                     OpenRB-150: policy_bridge.ino
                        |  Dynamixel bus: sync write/read 14x XL330
                        |  I2C: BNO055
                        ^  USB serial: state packet (74 B @ 50 Hz)
```

## Files
- `openrb_firmware/policy_bridge/policy_bridge.ino` — flash via Arduino IDE
- `jetson/robot_config.py` — joint order, defaults, scales (EDIT THIS FIRST)
- `jetson/run_policy.py` — the 50 Hz inference loop

## One-time setup checklist (in order)

1. **Servo baud rate**: every XL330 must be set to **1 Mbps** (factory default
   57600 is too slow for 50 Hz sync of 14 servos). Dynamixel Wizard 2.0, once
   per servo.

2. **Verify joint order** (critical): run the snippet in `robot_config.py`'s
   header on your desktop. PhysX orders joints breadth-first, not URDF order.
   Paste the real list into `JOINT_NAMES`.

3. **Fill the CALIB table** in the firmware: servo ID, sign, and zero-offset
   per joint, same index order as `JOINT_NAMES`. Reuse the sign conventions
   from your ros2_control hardware interface — you already debugged the
   mirrored right leg once.

4. **Verify against env.yaml**: `ACTION_SCALE` and the observation term list
   in `policies/<run>/params/env.yaml` are ground truth.

5. **Flash the firmware**, then on the Jetson:
   `pip install onnxruntime numpy pyserial` and check the port name
   (`ls /dev/ttyACM*`), update `SERIAL_PORT` if needed.

## Bring-up sequence (robot ON A STAND throughout)

1. **State stream test** — firmware flashed, `run_policy.py` NOT running:
   the board streams state anyway. `python3 -c` a quick read loop or just run
   `run_policy.py`; it prints when the stream is up.

2. **IMU sanity**: robot upright and still -> projected gravity ~ (0, 0, -1).
   Tilt forward (toward -Y) -> the Y component goes negative. If axes are
   swapped, fix `IMU_TO_BASE`. Rotate the board 90° in ~1 s and confirm gyro
   peaks near 1.5 rad/s (if it reads ~90, the gyro is in deg/s — set
   `GYRO_SCALE = PI/180` in the firmware).

3. **Joint sign check**: run the loop with all commands zero. Legs should
   settle into the crouch (hips/knees bent ~20°, feet level). A joint folding
   the wrong way = flip its `sign` in CALIB. This is the sim spawn-pose check,
   on hardware.

4. **Teleop on the stand**: press `w`. Legs should swing in a walking rhythm.
   v1 will look wrong (its base_lin_vel input is zeroed) — rhythmically
   plausible is the pass bar, walking quality comes with v2.

5. **Kill switch drill**: press `k`, confirm all servos go limp instantly.
   Unplug the USB mid-run, confirm the 200 ms watchdog does the same.

Only after all five: ground contact, tethered, with the v2 policy.

## Safety built in
- 200 ms watchdog: Jetson silent -> torque off (limp, not rigid)
- `k` = instant kill, ESC = kill + exit, Ctrl+C sends kill on the way out
- XOR checksums on every packet; corrupt packets are dropped, not executed
