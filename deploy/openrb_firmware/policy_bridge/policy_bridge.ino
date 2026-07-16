// ============================================================================
// policy_bridge.ino — OpenRB-150 "spinal cord" for AAndrade-Humanoid-Lite
// ============================================================================
// Role: dumb, fast, reliable. The Jetson runs the brain (RL policy); this
// board only:
//   1. Receives 14 joint position targets over USB serial (50 Hz)
//   2. Sync-writes them to the XL330s
//   3. Sync-reads present position+velocity, reads BNO055 quat+gyro
//   4. Streams a state packet back up
//   5. Torques OFF if the Jetson goes quiet (200 ms watchdog) or on KILL
//
// Libraries (Arduino IDE library manager):
//   - Dynamixel2Arduino
//   - Adafruit BNO055 (+ Adafruit Unified Sensor)
//
// SERIAL PROTOCOL (little-endian int16 throughout):
//   CMD   (Jetson->RB, 32 B): AA 55 01 | 14 x int16 target_milliradians | XOR
//   KILL  (Jetson->RB,  4 B): AA 55 02 | XOR
//   STATE (RB->Jetson, 74 B): AA 55 10 | 14 x int16 pos_mrad
//                                      | 14 x int16 vel_mrad_per_s
//                                      | 4  x int16 quat_w,x,y,z (x1e4)
//                                      | 3  x int16 gyro_mrad_per_s (x,y,z)
//                                      | XOR
//   XOR checksum covers every byte after the AA 55 header.
//
// Angle convention: milliradians in URDF joint space (0 = URDF zero pose).
// The CALIB table below converts URDF radians <-> Dynamixel ticks per joint.
// ============================================================================

#include <Dynamixel2Arduino.h>
#include <Adafruit_BNO055.h>
#include <Wire.h>

// ---------------------------------------------------------------------------
// CONFIG
// ---------------------------------------------------------------------------
#define DXL_SERIAL   Serial1
#define DXL_DIR_PIN  -1            // OpenRB-150: direction handled internally
const uint32_t DXL_BAUD = 1000000; // !!! All XL330s must be set to 1 Mbps.
                                   // 57600 (factory default) is too slow for
                                   // 50 Hz sync of 14 servos. Change each
                                   // servo's baud once via Dynamixel Wizard.
const float DXL_PROTOCOL = 2.0;

const uint8_t  NUM_JOINTS = 14;
const uint32_t STATE_PERIOD_US   = 20000;  // 50 Hz state stream
const uint32_t WATCHDOG_TIMEOUT_MS = 200;  // no CMD for this long -> torque off

// ---------------------------------------------------------------------------
// JOINT CALIBRATION TABLE  — THE critical sim-to-real mapping. VERIFY IT.
//
// Index order MUST match JOINT_NAMES in the Jetson's robot_config.py
// (which must match the policy's training order — see deploy README).
//
// For each joint:
//   id     : Dynamixel servo ID on the bus
//   sign   : +1/-1 so that a POSITIVE URDF joint angle moves the physical
//            joint the same direction as in simulation (your mirrored right
//            leg will need flipped signs — reuse the conventions from your
//            ros2_control hardware interface, you already solved this once)
//   offset : tick reading when the joint is at URDF zero (straight-leg pose).
//            2048 if you assembled every horn centered; measure otherwise.
// ---------------------------------------------------------------------------
struct JointCalib { uint8_t id; int8_t sign; int32_t offset; };

JointCalib CALIB[NUM_JOINTS] = {
  // TODO(anthony): set real IDs/signs/offsets. Order = robot_config.JOINT_NAMES
  { 1, +1, 2048},  //  0 left_hip_pitch
  { 8, -1, 2048},  //  1 right_hip_pitch
  { 2, +1, 2048},  //  2 left_hip_roll
  { 9, -1, 2048},  //  3 right_hip_roll
  { 3, +1, 2048},  //  4 left_leg_yaw
  {10, -1, 2048},  //  5 right_leg_yaw
  { 4, +1, 2048},  //  6 left_knee1_pitch
  {11, -1, 2048},  //  7 right_knee1_pitch
  { 5, +1, 2048},  //  8 left_knee2_pitch
  {12, -1, 2048},  //  9 right_knee2_pitch
  { 6, +1, 2048},  // 10 left_ankle_pitch
  {13, -1, 2048},  // 11 right_ankle_pitch
  { 7, +1, 2048},  // 12 left_ankle_roll
  {14, -1, 2048},  // 13 right_ankle_roll
};

const float TICKS_PER_RAD = 4096.0f / (2.0f * PI);          // XL330: 4096/rev
const float VEL_UNIT_RAD_S = 0.229f * 2.0f * PI / 60.0f;    // 0.229 rpm/unit
// Gyro scale: Adafruit_BNO055 VECTOR_GYROSCOPE nominally returns rad/s, but
// unit handling has known quirks. VERIFY: rotate the board ~90 deg in ~1 s
// and confirm the logged rate peaks near 1.5 rad/s. Adjust if off by ~57x
// (that means it's actually deg/s -> set to PI/180).
const float GYRO_SCALE = 1.0f;

// ---------------------------------------------------------------------------
Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28, &Wire);
using namespace ControlTableItem;

// Sync read: Present Velocity (addr 128, 4B) + Present Position (addr 132, 4B)
// are contiguous -> one 8-byte read per servo, one bus transaction total.
const uint16_t SR_ADDR = 128;
const uint16_t SR_LEN  = 8;
struct __attribute__((packed)) SrData { int32_t vel; int32_t pos; };
SrData    srData[NUM_JOINTS];
DYNAMIXEL::InfoSyncReadInst_t  srInfo;
DYNAMIXEL::XELInfoSyncRead_t   srXels[NUM_JOINTS];
uint8_t   srPktBuf[256];

// Sync write: Goal Position (addr 116, 4B)
const uint16_t SW_ADDR = 116;
const uint16_t SW_LEN  = 4;
int32_t   swTicks[NUM_JOINTS];
DYNAMIXEL::InfoSyncWriteInst_t swInfo;
DYNAMIXEL::XELInfoSyncWrite_t  swXels[NUM_JOINTS];

// State
int16_t  targetsMrad[NUM_JOINTS] = {0};
bool     torqueOn = false;
uint32_t lastCmdMs = 0;
uint32_t lastStateUs = 0;

// Serial RX parser
uint8_t rxBuf[64];
uint8_t rxLen = 0;

// ---------------------------------------------------------------------------
void setTorqueAll(bool on) {
  for (uint8_t i = 0; i < NUM_JOINTS; i++) {
    if (on) dxl.torqueOn(CALIB[i].id); else dxl.torqueOff(CALIB[i].id);
  }
  torqueOn = on;
}

void setup() {
  Serial.begin(115200);          // USB CDC: value ignored, always full speed

  dxl.begin(DXL_BAUD);
  dxl.setPortProtocolVersion(DXL_PROTOCOL);
  for (uint8_t i = 0; i < NUM_JOINTS; i++) {
    dxl.torqueOff(CALIB[i].id);
    dxl.setOperatingMode(CALIB[i].id, OP_POSITION);
  }

  // Sync read setup
  srInfo.packet.p_buf = srPktBuf;
  srInfo.packet.buf_capacity = sizeof(srPktBuf);
  srInfo.packet.is_completed = false;
  srInfo.addr = SR_ADDR;
  srInfo.addr_length = SR_LEN;
  srInfo.p_xels = srXels;
  srInfo.xel_count = NUM_JOINTS;
  for (uint8_t i = 0; i < NUM_JOINTS; i++) {
    srXels[i].id = CALIB[i].id;
    srXels[i].p_recv_buf = (uint8_t*)&srData[i];
  }
  srInfo.is_info_changed = true;

  // Sync write setup
  swInfo.packet.p_buf = nullptr;
  swInfo.packet.is_completed = false;
  swInfo.addr = SW_ADDR;
  swInfo.addr_length = SW_LEN;
  swInfo.p_xels = swXels;
  swInfo.xel_count = NUM_JOINTS;
  for (uint8_t i = 0; i < NUM_JOINTS; i++) {
    swXels[i].id = CALIB[i].id;
    swXels[i].p_data = (uint8_t*)&swTicks[i];
  }
  swInfo.is_info_changed = true;

  if (!bno.begin(OPERATION_MODE_NDOF)) {
    // IMU missing: halt with fast blink — do NOT run a blind robot.
    pinMode(LED_BUILTIN, OUTPUT);
    while (1) { digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN)); delay(100); }
  }
  bno.setExtCrystalUse(true);
}

// ---------------------------------------------------------------------------
void handleCmd(const uint8_t* payload) {
  memcpy(targetsMrad, payload, NUM_JOINTS * 2);
  lastCmdMs = millis();
  if (!torqueOn) setTorqueAll(true);   // first command arms the servos

  for (uint8_t i = 0; i < NUM_JOINTS; i++) {
    float rad = targetsMrad[i] / 1000.0f;
    swTicks[i] = CALIB[i].offset + (int32_t)(CALIB[i].sign * rad * TICKS_PER_RAD);
  }
  swInfo.is_info_changed = true;
  dxl.syncWrite(&swInfo);
}

void parseSerial() {
  while (Serial.available()) {
    uint8_t b = Serial.read();
    if (rxLen == 0 && b != 0xAA) continue;
    if (rxLen == 1 && b != 0x55) { rxLen = 0; continue; }
    rxBuf[rxLen++] = b;

    if (rxLen >= 3) {
      uint8_t type = rxBuf[2];
      uint8_t need = (type == 0x01) ? 32 : (type == 0x02) ? 4 : 0;
      if (need == 0) { rxLen = 0; continue; }          // unknown type
      if (rxLen == need) {
        uint8_t x = 0;
        for (uint8_t i = 2; i < need - 1; i++) x ^= rxBuf[i];
        if (x == rxBuf[need - 1]) {
          if (type == 0x01) handleCmd(&rxBuf[3]);
          else if (type == 0x02) setTorqueAll(false);  // KILL
        }
        rxLen = 0;
      }
    }
    if (rxLen >= sizeof(rxBuf)) rxLen = 0;
  }
}

void sendState() {
  srInfo.is_info_changed = true;
  dxl.syncRead(&srInfo);

  imu::Quaternion q = bno.getQuat();
  imu::Vector<3> g = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);

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
  p[28] = (int16_t)(q.w() * 10000.0);                  // quat x1e4
  p[29] = (int16_t)(q.x() * 10000.0);
  p[30] = (int16_t)(q.y() * 10000.0);
  p[31] = (int16_t)(q.z() * 10000.0);
  p[32] = (int16_t)(g.x() * GYRO_SCALE * 1000.0);      // gyro mrad/s
  p[33] = (int16_t)(g.y() * GYRO_SCALE * 1000.0);
  p[34] = (int16_t)(g.z() * GYRO_SCALE * 1000.0);

  uint8_t x = 0;
  for (uint8_t i = 2; i < 73; i++) x ^= pkt[i];
  pkt[73] = x;
  Serial.write(pkt, 74);
}

void loop() {
  parseSerial();

  // Watchdog: Jetson silent -> torque off, robot goes limp instead of rigid
  if (torqueOn && (millis() - lastCmdMs > WATCHDOG_TIMEOUT_MS)) {
    setTorqueAll(false);
  }

  uint32_t now = micros();
  if (now - lastStateUs >= STATE_PERIOD_US) {
    lastStateUs = now;
    sendState();
  }
}
