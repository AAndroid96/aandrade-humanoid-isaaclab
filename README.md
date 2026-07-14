# AAndrade-Humanoid-Lite — Isaac Lab Training Package

Velocity-tracking locomotion policy for the 14-DOF biped (XL330-M288-T ×14,
OpenRB-150, BNO055), following the same structure as the Unitree H1/G1
examples in `isaaclab_tasks`. Written for **Isaac Lab 2.x** (`isaaclab`
namespace).

## What was wrong with the URDF (and what got fixed)

`robot_isaac.urdf` is your uploaded URDF, patched by `scripts/patch_urdf.py`:

1. **All 15 links had mass = 1e-09 kg** (total: 15 nanograms). onshape-to-robot
   exports placeholder inertials when parts have no material assigned in
   Onshape. The patch assigns *estimated* masses (0.72 kg total: 180 g pelvis,
   40 g leg links, 30 g feet) with solid-box inertias.
   **Strongly recommended:** assign materials in Onshape and re-export so the
   inertias come from real CAD mass properties — then re-run the patch script
   with `MASS_ESTIMATES` emptied (it will still fix limits and mesh paths).
2. **effort=10 N·m / velocity=10 rad/s** joint limits — ~20× the XL330-M288's
   real stall torque. Patched to 0.52 N·m / 10.5 rad/s.
3. **`package://assets/...` mesh paths** → rewritten to relative `assets/...`
   so the Isaac Lab URDF importer resolves them. Keep the URDF next to your
   `assets/` STL folder.

## Workflow

### 1. Convert URDF → USD

Put `robot_isaac.urdf` in the same directory as your `assets/` folder, then:

```bash
cd ~/IsaacLab
./isaaclab.sh -p scripts/tools/convert_urdf.py \
  ~/robots/aandrade_humanoid/robot_isaac.urdf \
  ~/robots/aandrade_humanoid/robot_isaac.usd \
  --merge-joints --joint-stiffness 0.0 --joint-damping 0.0 --joint-target-type none
```

Stiffness/damping are zeroed at import because the `DCMotorCfg` actuator
model applies PD gains at runtime.

### 2. Install this package

```bash
export AANDRADE_USD_PATH=~/robots/aandrade_humanoid/robot_isaac.usd
cd ~/IsaacLab
./isaaclab.sh -p -m pip install -e /path/to/aandrade_isaaclab/source/aandrade_humanoid
```

### 3. VERIFY THE SPAWN POSE BEFORE TRAINING

This is the step that catches the mirrored-joint sign issue — the same class
of bug you hit in `robot_balance`. The default joint positions in
`robots/aandrade_humanoid.py` use mirrored signs derived from the URDF limits
(left knee1 ∈ [0, +π/2], right knee1 ∈ [−π/2, 0], etc.), but the crouch
values are educated guesses.

```bash
./isaaclab.sh -p scripts/environments/zero_agent.py \
  --task Isaac-Velocity-Flat-AAndrade-Humanoid-v0 --num_envs 4
```

Check in the viewport: symmetric crouch, feet flat, pelvis at spawn height.
If a leg folds backwards, flip the sign for that joint pair in
`init_state.joint_pos`. Also adjust `pos=(0, 0, 0.24)` so the feet just
touch the ground — measure the actual standing pelvis height from the USD.

### 4. Train

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-AAndrade-Humanoid-v0 \
  --num_envs 4096 --headless
```

Your 3080 Ti (12 GB) should handle 4096 envs for a 14-DOF robot; drop to
2048 if you hit VRAM limits. Expect a few thousand iterations (~1-3 h).

### 5. Play / evaluate

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-AAndrade-Humanoid-Play-v0 --num_envs 32
```

## Tuning knobs (in rough priority order)

1. **Default pose + spawn height** — nothing works until this is right.
2. **PD gains** (`stiffness=2.0, damping=0.08`) — the robot should hold its
   crouch under zero actions without oscillating or collapsing. If it
   vibrates, lower stiffness or raise damping; if it sags, raise stiffness.
3. **`armature=0.002`** — reflected rotor inertia for 288:1 gearing. This
   matters a lot for sim-to-real on small geared servos; consider system ID
   later (log a chirp on one servo via the OpenRB-150 and fit).
4. **Command ranges** — start small (already scaled: max 0.3 m/s forward).
5. **Reward weights** — `feet_air_time` threshold (0.3 s) sets gait cadence;
   small robots step faster, so try 0.2 s if steps look floaty.

## Sim-to-real notes for your pipeline

- The policy outputs joint position targets; on hardware, map them to
  XL330 position-control goals at the same control rate (template runs
  policy at 50 Hz — matches your existing 50 Hz Arduino loop).
- Add domain randomization on friction/gains before deploying (the template
  already randomizes mass and pushes; physics material randomization is in
  `events`).
- On the Jetson Orin Nano, export the policy with the `play.py --export`
  path (ONNX/TorchScript) and run it in your ros2_control chain.

## Known caveats

- Link names are onshape-to-robot artifacts (`servobody_4` = left foot,
  `servobody_8` = right foot). Consider a `config.json` link-renaming pass
  in onshape-to-robot on your next export; then update `FEET`, `PELVIS`,
  and `UPPER_LEG_BODIES` in `velocity_env_cfg.py`.
- Estimated inertias are placeholders. Real mass properties from Onshape
  will noticeably change PD tuning and learned gaits.
- If you're on Isaac Lab 1.x, replace `isaaclab.*` imports with
  `omni.isaac.lab.*` and `isaaclab_tasks`/`isaaclab_rl` with their
  `omni.isaac.lab_tasks` equivalents.
