#include "my_functions.h"

/***********************************************************
 * setMotors
 */

const float WHEEL_DIAMETER_MM = 68.0;
//const float GEAR_RATIO        = 30.0;
//const float ENCODER_BASE_PPR  = (693 + 696)/2;
const float TRACK_WIDTH_MM    = 317;

// --- Calculated Constants ---
const float TICKS_PER_REVOLUTION  = 2340.0;
const float DISTANCE_PER_TICK = M_PI * WHEEL_DIAMETER_MM / TICKS_PER_REVOLUTION;


// --- Tracking Variables ---
int16_t prev_ticks_left  = 0;
int16_t prev_ticks_right = 0;
float total_distance = 0.0;
float current_angle  = 0.0;
float robot_x = 0.0;
float robot_y = 0.0;

// --- Timing ---
unsigned long last_loop_time = 0;
const unsigned long LOOP_INTERVAL_MS = 100;

// --- PID ---
const float Kp          = 0.3;   // tune this: increase if still drifting, decrease if wobbling was 2.2
const int   BASE_SPEED  = 200;  // was 180
int16_t pid_ticks_left  = 0;   // tick snapshot when movement started
int16_t pid_ticks_right = 0;
float integral = 0.0;
const float Ki = 0.5;
bool pid_active = false;
int pid_startup_count = 0;
const int PID_STARTUP_IGNORE = 3;  // ignore first 3 readings

// --- Web Server & WebSocket ---
const char *ssid     = "ESP32-AP";
const char *password = "LetMeInPlz";
const int http_port  = 80;
const int ws_port    = 1337;

//AsyncWebServer   server(80);

char msg_buf[10];

// --- MPU6050 ---
MPU6050 mpu;
float gyro_offset_z = 0;
float heading_deg = 0;
unsigned long last_gyro_time = 0;

// --- Servo ---
const int MIN_ANGLE = 0;     // Semicircle start angle
const int MAX_ANGLE = 180;   // Semicircle end angle
const int STEP_DELAY = 15;
int current_servo_angle;
bool sweepForward = true;
float last_scan_x = 0.0;
float last_scan_y = 0.0;
const float MIN_RESCAN_DISTANCE = 150.0;  // mm — tune this
const int x_offset = 54.0;
const int y_offset = 0;
const int SERVO_STEP_SIZE = 10;          // was 5 — fewer, more deliberate steps
const unsigned long SERVO_SETTLE_MS = 120;  // tune this: raise if still jittery, lower if scans feel slow
bool waiting_for_settle = false;
unsigned long servo_step_time = 0;

// --- Sweep Scan Data Array ---
uint16_t sweep_distances[SWEEP_SAMPLES];
bool is_sweeping = false;
bool perm_to_scan = true;
uint16_t scan_array[ServoArraySize] = {0};

void setMotors(int left_speed, int right_speed) {
  

  if (left_speed == 0 && right_speed == 0) {
    // STOP
    pid_active = false;
    ledcWrite(leftChannel,  0);
    ledcWrite(rightChannel, 0);
    digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
    digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
    return;
  }

  if (left_speed > 0 && right_speed > 0) {
    // FORWARD — enable PID, snapshot current ticks
    pid_startup_count = 0;
    integral = 0.0;
    pid_active = true;
    int16_t tl = 0, tr = 0;
    pcnt_get_counter_value(PCNT_UNIT_0, &tl);
    pcnt_get_counter_value(PCNT_UNIT_1, &tr);
    pid_ticks_left  = tl;
    pid_ticks_right = tr;
    ledcWrite(leftChannel,  BASE_SPEED);
    ledcWrite(rightChannel, BASE_SPEED);
    digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH);
    digitalWrite(IN3, LOW);  digitalWrite(IN4, HIGH);

  } else if (left_speed < 0 && right_speed < 0) {
    // BACKWARD — enable PID
    pid_startup_count = 0;
    integral = 0.0;
    pid_active = true;
    int16_t tl = 0, tr = 0;
    pcnt_get_counter_value(PCNT_UNIT_0, &tl);
    pcnt_get_counter_value(PCNT_UNIT_1, &tr);
    pid_ticks_left  = tl;
    pid_ticks_right = tr;
    ledcWrite(leftChannel,  BASE_SPEED);
    ledcWrite(rightChannel, BASE_SPEED);
    digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
    digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);

  } else if (left_speed < 0 && right_speed > 0) {
    // TURN LEFT — no PID
    pid_active = false;
    ledcWrite(leftChannel,  BASE_SPEED);
    ledcWrite(rightChannel, BASE_SPEED);
    digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
    digitalWrite(IN3, HIGH);  digitalWrite(IN4, LOW);

  } else if (left_speed > 0 && right_speed < 0) {
    // TURN RIGHT — no PID
    pid_active = false;
    ledcWrite(leftChannel,  BASE_SPEED);
    ledcWrite(rightChannel, BASE_SPEED);
    digitalWrite(IN1, HIGH);  digitalWrite(IN2, LOW);
    digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
  }
}


// Handler for your outgoing Python connection (The new code)
void onPythonEvent(WStype_t type, uint8_t *payload, size_t length) {
    switch(type) {
        case WStype_DISCONNECTED:
            Serial.println("[Python Link] 🔴 Disconnected from Python Server.");
            break;
            
        case WStype_CONNECTED:
            Serial.println("[Python Link] 🟢 Connected! Sending initial check-in...");
            // Send an immediate notification packet upon connecting
            webSocketClient.sendTXT("ESP32_READY");
            break;
            
        case WStype_TEXT:
            //Serial.printf("Received: %s\n", payload);
            if      (strcmp((char *)payload, "FORWARD")  == 0) setMotors( 255,  255);
            else if (strcmp((char *)payload, "BACKWARD") == 0) setMotors(-255, -255);
            else if (strcmp((char *)payload, "LEFT")     == 0) setMotors(-255,  255);
            else if (strcmp((char *)payload, "RIGHT")    == 0) setMotors( 255, -255);
            else if (strcmp((char *)payload, "STOP")     == 0) setMotors(0, 0);
            else if (strcmp((char *)payload, "RESET") == 0) {
                robot_x = 0.0;
                robot_y = 0.0;
                current_angle = 0.0;
                total_distance = 0.0;
                prev_ticks_left  = 0;
                prev_ticks_right = 0;
                pcnt_counter_clear(PCNT_UNIT_0);
                pcnt_counter_clear(PCNT_UNIT_1);
                Serial.println("Origin reset");
            }
            break;
    }
}

/***********************************************************
 * HTTP callbacks
 */
void onIndexRequest(AsyncWebServerRequest *request) {
  Serial.println("[HTTP] GET /");
  request->send(SPIFFS, "/index.html", "text/html");
}
void onCSSRequest(AsyncWebServerRequest *request) {
  Serial.println("[HTTP] GET /style.css");
  request->send(SPIFFS, "/style.css", "text/css");
}
void onPageNotFound(AsyncWebServerRequest *request) {
  request->send(404, "text/plain", "Not found");
}

/***********************************************************
 * PCNT init
 */
void init_hardware_pcnt(pcnt_unit_t unit, int gpio_a, int gpio_b) {
  pcnt_config_t cfg = {
    .pulse_gpio_num = gpio_a,
    .ctrl_gpio_num  = gpio_b,
    .lctrl_mode     = PCNT_MODE_REVERSE,
    .hctrl_mode     = PCNT_MODE_KEEP,
    .pos_mode       = PCNT_COUNT_INC,
    .neg_mode       = PCNT_COUNT_DEC,
    .counter_h_lim  = 32767,
    .counter_l_lim  = -32768,
    .unit           = unit,
    .channel        = PCNT_CHANNEL_0,
  };
  pcnt_unit_config(&cfg);
  pcnt_set_filter_value(unit, 10);
  pcnt_filter_enable(unit);
  pcnt_counter_clear(unit);
  pcnt_counter_resume(unit);
}

/***********************************************************
 * calibrateMPU
 */
void calibrateMPU() {
    Serial.println("Keep robot still...");

    long sum = 0;
    const int samples = 2000;

    for (int i = 0; i < samples; i++) {
        int16_t ax, ay, az, gx, gy, gz;
        mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

        sum += gz;
        delay(2);
        yield(); 
    }

    gyro_offset_z = (float)sum / samples;

    Serial.print("gyro_offset_z = ");
    Serial.println(gyro_offset_z);
}

