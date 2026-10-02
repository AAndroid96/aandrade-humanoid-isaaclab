// ============================================================================
// policy_bridge.ino — OpenRB-150 "spinal cord" for AAndrade-Humanoid-Lite
// ============================================================================
// v4 (2026-09-21). The design rule for this version: THE BOARD MUST ALWAYS
// REACH loop() AND ALWAYS STREAM. Every previous version could stall or
// crash before producing a single byte, which made the whole system opaque.
//
// Fixes vs v3:
//   * NO Dynamixel I/O IN setup(). v1-v3 called torqueOff + setOperatingMode
//     on 14 servos in setup(); with an unpowered bus every one of those
//     timed out and the board never reached loop(). Setup is now pure
//     memory work and returns in microseconds.
//   * SERVO DISCOVERY IS INCREMENTAL. loop() pings ONE servo per cycle,
//     then configures ONE per cycle. A dead bus costs one timeout per
//     cycle instead of 28 back to back, and the state stream runs the
//     whole time.
//   * syncRead IS SKIPPED when no servos answered, so a bare board runs at
//     full speed with no bus timeouts at all.
//   * Sync-read buffer 256 -> 512 B. 14 servos return ~266 B of status
//     packets (19 B each); the old buffer overran memory on the first read
//     and hard-faulted the chip, killing USB mid-enumeration. THIS was the
//     bug behind months of "the board never streams anything".
//
// LED CODE (readable with no serial connection at all):
//   fast blink,  ~10 Hz  -> discovering servos
//   slow blink,  ~0.5 Hz -> running, NO servos found (bare board / no power)
//   medium blink, ~3 Hz  -> running, servos found and configured
//   frozen               -> crashed. Should not happen in v4.
//
// SERIAL PROTOCOL (little-endian int16) — unchanged since v1:
//   CMD   (Jetson->RB, 32 B): AA 55 01 | 14 x int16 target_milliradians | XOR
//   KILL  (Jetson->RB,  4 B): AA 55 02 | XOR
//   STATE (RB->Jetson, 74 B): AA 55 10 | 14 x int16 pos_mrad
//                                      | 14 x int16 vel_mrad_per_s
//                                      | 4  x int16 quat (always 1,0,0,0)
//                                      | 3  x int16 gyro (always 0,0,0)
//                                      | XOR
//   quat/gyro are placeholders; the Jetson reads the real BNO055 itself.
//
// POWER WARNING: 14 XL330s cannot run from a USB port. Feed the OpenRB's DC
// input from a regulated 5 V supply rated 5 A or more. NEVER 12 V — XL330s
// are 5 V parts with a 6 V ceiling and 12 V destroys them.
// ============================================================================

#include <Dynamixel2Arduino.h>

// ---------------------------------------------------------------------------
// CONFIG
// ---------------------------------------------------------------------------
#define DXL_SERIAL   Serial1
#define DXL_DIR_PIN  -1            // OpenRB-150: direction handled internally
const uint32_t DXL_BAUD = 1000000; // All XL330s set to 1 Mbps (2026-07)
const float DXL_PROTOCOL = 2.0;

const uint8_t  NUM_JOINTS = 14;
const uint32_t STATE_PERIOD_US     = 20000;  // 50 Hz state stream
const uint32_t WATCHDOG_TIMEOUT_MS = 200;    // no CMD this long -> torque off

const uint32_t BEAT_DISCOVER_MS = 50;    // ~10 Hz  : discovering
const uint32_t BEAT_NO_SERVOS_MS = 1000; // ~0.5 Hz : running, no servos
const uint32_t BEAT_READY_MS = 150;      // ~3 Hz   : running, servos found

// ---------------------------------------------------------------------------
// JOINT CALIBRATION TABLE — index order MUST match JOINT_NAMES in
// robot_config.py (verified against the trained env 2026-07-26).
//   id     : Dynamixel servo ID (left leg 1-7, right leg 11-17,
//            hip_pitch -> ankle_roll down the chain)
//   sign   : +1/-1 so a POSITIVE URDF angle moves the joint the same way as
//            in sim. Validate on the stand; flip any joint that folds wrong.
//   offset : tick reading at URDF zero (straight-leg pose).
// ---------------------------------------------------------------------------
struct JointCalib { uint8_t id; int8_t sign; int32_t offset; };

JointCalib CALIB[NUM_JOINTS] = {
  { 1, +1, 2048},  //  0 left_hip_pitch
  {11, -1, 2048},  //  1 right_hip_pitch
  { 2, +1, 2048},  //  2 left_hip_roll
  {12, -1, 2048},  //  3 right_hip_roll
  { 3, +1, 2048},  //  4 left_leg_yaw
  {13, -1, 2048},  //  5 right_leg_yaw
  { 4, +1, 2048},  //  6 left_knee1_pitch
  {14, -1, 2048},  //  7 right_knee1_pitch
  { 5, +1, 2048},  //  8 left_knee2_pitch
  {15, -1, 2048},  //  9 right_knee2_pitch
  { 6, +1, 2048},  // 10 left_ankle_pitch
  {16, -1, 2048},  // 11 right_ankle_pitch
  { 7, +1, 2048},  // 12 left_ankle_roll
  {17, -1, 2048},  // 13 right_ankle_roll
};

const float TICKS_PER_RAD  = 4096.0f / (2.0f * PI);       // XL330: 4096/rev
const float VEL_UNIT_RAD_S = 0.229f * 2.0f * PI / 60.0f;  // 0.229 rpm/unit

// ---------------------------------------------------------------------------
Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);
using namespace ControlTableItem;

// Sync read: Present Velocity (128, 4B) + Present Position (132, 4B) are
// contiguous -> one 8-byte read per servo in a single bus transaction.
const uint16_t SR_ADDR = 128;
const uint16_t SR_LEN  = 8;
struct __attribute__((packed)) SrData { int32_t vel; int32_t pos; };
SrData    srData[NUM_JOINTS];
DYNAMIXEL::InfoSyncReadInst_t  srInfo;
DYNAMIXEL::XELInfoSyncRead_t   srXels[NUM_JOINTS];
uint8_t   srPktBuf[512];     // 14 x 19 B = 266 B needed. 256 was the old bug.

// Sync write: Goal Position (116, 4B)
const uint16_t SW_ADDR = 116;
const uint16_t SW_LEN  = 4;
int32_t   swTicks[NUM_JOINTS];
DYNAMIXEL::InfoSyncWriteInst_t swInfo;
DYNAMIXEL::XELInfoSyncWrite_t  swXels[NUM_JOINTS];

// --- discovery state machine ------------------------------------------------
enum InitPhase : uint8_t { PHASE_PING, PHASE_CONFIG, PHASE_READY };
InitPhase phase = PHASE_PING;
uint8_t   phaseIndex = 0;
bool      servoPresent[NUM_JOINTS] = {false};
uint8_t   servoCount = 0;

// --- runtime state ----------------------------------------------------------
int16_t  targetsMrad[NUM_JOINTS] = {0};
bool     torqueOn = false;
uint32_t lastCmdMs = 0;
uint32_t lastStateUs = 0;
uint32_t lastBeatMs = 0;
bool     ledState = false;

uint8_t  rxBuf[64];
uint8_t  rxLen = 0;

// ---------------------------------------------------------------------------
void setTorqueAll(bool on) {
  for (uint8_t i = 0; i < NUM_JOINTS; i++) {
    if (!servoPresent[i]) continue;             // never talk to absent servos
    if (on) dxl.torqueOn(CALIB[i].id); else dxl.torqueOff(CALIB[i].id);
  }
  torqueOn = on;
}

// One unit of discovery work per call. Never more than a single bus timeout.
void stepInit() {
  if (phase == PHASE_PING) {
    if (dxl.ping(CALIB[phaseIndex].id)) {
      servoPresent[phaseIndex] = true;
      servoCount++;
    }
    if (++phaseIndex >= NUM_JOINTS) { phase = PHASE_CONFIG; phaseIndex = 0; }
    return;
  }

  if (phase == PHASE_CONFIG) {
    uint8_t i = phaseIndex;
    if (servoPresent[i]) {
      dxl.torqueOff(CALIB[i].id);
      dxl.setOperatingMode(CALIB[i].id, OP_POSITION);
    }
    if (++phaseIndex >= NUM_JOINTS) {
      // Build the sync lists from whatever actually answered.
      uint8_t n = 0;
      for (uint8_t j = 0; j < NUM_JOINTS; j++) {
        if (!servoPresent[j]) continue;
        srXels[n].id = CALIB[j].id;
        srXels[n].p_recv_buf = (uint8_t*)&srData[j];
        swXels[n].id = CALIB[j].id;
        swXels[n].p_data = (uint8_t*)&swTicks[j];
        n++;
      }
      srInfo.xel_count = n;
      swInfo.xel_count = n;
      srInfo.is_info_changed = true;
      swInfo.is_info_changed = true;
      phase = PHASE_READY;
    }
    return;
  }
}

// ---------------------------------------------------------------------------
void setup() {
  SerialUSB.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  // Port setup only — no servo transactions here. This is what guarantees
  // the board reaches loop() and starts streaming no matter what is (or
  // is not) attached to the Dynamixel bus.
  dxl.begin(DXL_BAUD);
  dxl.setPortProtocolVersion(DXL_PROTOCOL);

  srInfo.packet.p_buf = srPktBuf;
  srInfo.packet.buf_capacity = sizeof(srPktBuf);
  srInfo.packet.is_completed = false;
  srInfo.addr = SR_ADDR;
  srInfo.addr_length = SR_LEN;
  srInfo.p_xels = srXels;
  srInfo.xel_count = 0;                      // filled in after discovery

  swInfo.packet.p_buf = nullptr;             // library's internal buffer
  swInfo.packet.is_completed = false;
  swInfo.addr = SW_ADDR;
  swInfo.addr_length = SW_LEN;
  swInfo.p_xels = swXels;
  swInfo.xel_count = 0;

  for (uint8_t i = 0; i < NUM_JOINTS; i++) { // sane defaults = "at URDF zero"
    srData[i].pos = CALIB[i].offset;
    srData[i].vel = 0;
    swTicks[i] = CALIB[i].offset;
  }

  lastBeatMs = millis();
}

// ---------------------------------------------------------------------------
void handleCmd(const uint8_t* payload) {
  memcpy(targetsMrad, payload, NUM_JOINTS * 2);
  lastCmdMs = millis();
  if (!torqueOn && servoCount > 0) setTorqueAll(true);   // arm on first cmd

  for (uint8_t i = 0; i < NUM_JOINTS; i++) {
    float rad = targetsMrad[i] / 1000.0f;
    swTicks[i] = CALIB[i].offset + (int32_t)(CALIB[i].sign * rad * TICKS_PER_RAD);
  }
  if (servoCount > 0) {
    swInfo.is_info_changed = true;
    dxl.syncWrite(&swInfo);
  }
}

void parseSerial() {
  while (SerialUSB.available()) {
    uint8_t b = SerialUSB.read();
    if (rxLen == 0 && b != 0xAA) continue;
    if (rxLen == 1 && b != 0x55) { rxLen = 0; continue; }
    rxBuf[rxLen++] = b;

    if (rxLen >= 3) {
      uint8_t type = rxBuf[2];
      uint8_t need = (type == 0x01) ? 32 : (type == 0x02) ? 4 : 0;
      if (need == 0) { rxLen = 0; continue; }
      if (rxLen == need) {
        uint8_t x = 0;
        for (uint8_t i = 2; i < need - 1; i++) x ^= rxBuf[i];
        if (x == rxBuf[need - 1]) {
          if (type == 0x01) handleCmd(&rxBuf[3]);
          else if (type == 0x02) setTorqueAll(false);     // KILL
        }
        rxLen = 0;
      }
    }
    if (rxLen >= sizeof(rxBuf)) rxLen = 0;
  }
}

void sendState() {
  if (servoCount > 0) {          // skip the bus entirely on a bare board
    srInfo.is_info_changed = true;
    dxl.syncRead(&srInfo);
  }

  uint8_t pkt[74];
  pkt[0] = 0xAA; pkt[1] = 0x55; pkt[2] = 0x10;
  int16_t* p = (int16_t*)&pkt[3];
  for (uint8_t i = 0; i < NUM_JOINTS; i++) {           // pos, mrad
    float rad = CALIB[i].sign * (srData[i].pos - CALIB[i].offset) / TICKS_PER_RAD;
    p[i] = (int16_t)(rad * 1000.0f);
  }
  for (uint8_t i = 0; i < NUM_JOINTS; i++) {           // vel, mrad/s
    float rs = CALIB[i].sign * srData[i].vel * VEL_UNIT_RAD_S;
    p[14 + i] = (int16_t)(rs * 1000.0f);
  }
  p[28] = 10000; p[29] = 0; p[30] = 0; p[31] = 0;      // quat identity
  p[32] = 0;     p[33] = 0; p[34] = 0;                 // gyro zero

  uint8_t x = 0;
  for (uint8_t i = 2; i < 73; i++) x ^= pkt[i];
  pkt[73] = x;
  SerialUSB.write(pkt, 74);
}

void loop() {
  parseSerial();

  // Watchdog: Jetson silent -> torque off, so the robot goes limp, not rigid
  if (torqueOn && (millis() - lastCmdMs > WATCHDOG_TIMEOUT_MS)) {
    setTorqueAll(false);
  }

  // Heartbeat — the LED encodes which state the firmware is in
  uint32_t beatPeriod = (phase != PHASE_READY) ? BEAT_DISCOVER_MS
                      : (servoCount == 0)      ? BEAT_NO_SERVOS_MS
                                               : BEAT_READY_MS;
  uint32_t nowMs = millis();
  if (nowMs - lastBeatMs >= beatPeriod) {
    lastBeatMs = nowMs;
    ledState = !ledState;
    digitalWrite(LED_BUILTIN, ledState);
  }

  // One unit of servo discovery per cycle — bounded, never a long stall
  if (phase != PHASE_READY) stepInit();

  // State stream runs from the very first cycle, discovery or not
  uint32_t now = micros();
  if (now - lastStateUs >= STATE_PERIOD_US) {
    lastStateUs = now;
    sendState();
  }
}
