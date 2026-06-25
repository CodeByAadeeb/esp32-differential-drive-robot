#include "driver/pcnt.h"
#include <Wire.h>
#include <MPU6050.h>
#include <WiFi.h>
#include <SPIFFS.h>
#include <ESPAsyncWebServer.h>
#include <WebSocketsServer.h>
#include <MPU6050.h>

// --- Hardware Configuration ---
#define LEFT_ENCODER_A_PIN  32
#define LEFT_ENCODER_B_PIN  5
#define RIGHT_ENCODER_A_PIN 19
#define RIGHT_ENCODER_B_PIN 18
#define ENA 12
#define IN1 14
#define IN2 27
#define ENB 26
#define IN3 25
#define IN4 33
#define frequency   1000
#define resolution  8
#define leftChannel  0
#define rightChannel 1

// --- Robot Constants ---
const float WHEEL_DIAMETER_MM = 43.0;
const float GEAR_RATIO        = 30.0;
const float ENCODER_BASE_PPR  = (693 + 696)/2;
const float TRACK_WIDTH_MM    = 122.0;

// --- Calculated Constants ---
float TICKS_PER_REVOLUTION;
float DISTANCE_PER_TICK;

// --- Tracking Variables ---
int16_t prev_ticks_left  = 0;
int16_t prev_ticks_right = 0;
float total_distance = 0.0;
float current_angle  = 0.0;
float robot_x = 0.0;
float robot_y = 0.0;

// --- Timing ---
unsigned long last_loop_time = 0;
const unsigned long LOOP_INTERVAL_MS = 20;

// --- PID ---
const float Kp          = 2.0;   // tune this: increase if still drifting, decrease if wobbling
const int   BASE_SPEED  = 180;
int16_t pid_ticks_left  = 0;   // tick snapshot when movement started
int16_t pid_ticks_right = 0;
float integral = 0.0;
const float Ki = 0.3;
bool pid_active = false;
int pid_startup_count = 0;
const int PID_STARTUP_IGNORE = 3;  // ignore first 3 readings

// --- Web Server & WebSocket ---
const char *ssid     = "ESP32-AP";
const char *password = "LetMeInPlz";
const int http_port  = 80;
const int ws_port    = 1337;

AsyncWebServer   server(80);
WebSocketsServer webSocket = WebSocketsServer(1337);
char msg_buf[10];

// --- MPU6050 ---
MPU6050 mpu;
float gyro_offset_z = 0;
float heading_deg = 0;
unsigned long last_gyro_time = 0;

/***********************************************************
 * Forward declarations
 */
void setMotors(int left_speed, int right_speed);

/***********************************************************
 * WebSocket callback
 */
void onWebSocketEvent(uint8_t client_num, WStype_t type,
                      uint8_t *payload, size_t length) {
  switch (type) {

    case WStype_DISCONNECTED:
      Serial.printf("[%u] Disconnected!\n", client_num);
      setMotors(0, 0);
      break;

    case WStype_CONNECTED: {
      IPAddress ip = webSocket.remoteIP(client_num);
      Serial.printf("[%u] Connection from ", client_num);
      Serial.println(ip.toString());
      break;
    }

    case WStype_TEXT:
      Serial.printf("[%u] Received: %s\n", client_num, payload);
      if      (strcmp((char *)payload, "FORWARD")  == 0) setMotors( 255,  255);
      else if (strcmp((char *)payload, "BACKWARD") == 0) setMotors(-255, -255);
      else if (strcmp((char *)payload, "LEFT")     == 0) setMotors(-255,  255);
      else if (strcmp((char *)payload, "RIGHT")    == 0) setMotors( 255, -255);
      else if (strcmp((char *)payload, "STOP")     == 0) setMotors(0, 0);
      break;

    default: break;
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
  pcnt_set_filter_value(unit, 100);
  pcnt_filter_enable(unit);
  pcnt_counter_clear(unit);
  pcnt_counter_resume(unit);
}

/***********************************************************
 * setMotors
 */
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
    digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
    digitalWrite(IN3, LOW);  digitalWrite(IN4, HIGH);

  } else if (left_speed > 0 && right_speed < 0) {
    // TURN RIGHT — no PID
    pid_active = false;
    ledcWrite(leftChannel,  BASE_SPEED);
    ledcWrite(rightChannel, BASE_SPEED);
    digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH);
    digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
  }
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
    }

    gyro_offset_z = (float)sum / samples;

    Serial.print("gyro_offset_z = ");
    Serial.println(gyro_offset_z);
}

/***********************************************************
 * setup
 */
void setup() {
  Serial.begin(115200);

  pinMode(ENA, OUTPUT); pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT); pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);

  ledcSetup(leftChannel,  frequency, resolution);
  ledcSetup(rightChannel, frequency, resolution);
  ledcAttachPin(ENA, leftChannel);
  ledcAttachPin(ENB, rightChannel);

  if (!SPIFFS.begin()) {
    Serial.println("Error mounting SPIFFS");
    while (1);
  }

  WiFi.softAP(ssid, password);
  Serial.println("\nAP running");
  Serial.print("My IP address: ");
  Serial.println(WiFi.softAPIP());

  server.on("/",          HTTP_GET, onIndexRequest);
  server.on("/style.css", HTTP_GET, onCSSRequest);
  server.onNotFound(onPageNotFound);
  server.begin();

  webSocket.begin();
  webSocket.onEvent(onWebSocketEvent);

  TICKS_PER_REVOLUTION = ENCODER_BASE_PPR * GEAR_RATIO;
  DISTANCE_PER_TICK    = (PI * WHEEL_DIAMETER_MM) / TICKS_PER_REVOLUTION;

  init_hardware_pcnt(PCNT_UNIT_0, LEFT_ENCODER_A_PIN,  LEFT_ENCODER_B_PIN);
  init_hardware_pcnt(PCNT_UNIT_1, RIGHT_ENCODER_A_PIN, RIGHT_ENCODER_B_PIN);

  Wire.begin(21, 23);

  mpu.initialize();

  calibrateMPU();

  last_gyro_time = millis();

  Serial.println("Hardware PCNT Odometry Initialized.");
}

/***********************************************************
 * loop
 */
void loop() {
  webSocket.loop();

  unsigned long current_time = millis();
  if (current_time - last_loop_time >= LOOP_INTERVAL_MS) {
    last_loop_time = current_time;

    // Read ticks
    int16_t current_ticks_left  = 0;
    int16_t current_ticks_right = 0;
    pcnt_get_counter_value(PCNT_UNIT_0, &current_ticks_left);
    pcnt_get_counter_value(PCNT_UNIT_1, &current_ticks_right);

    // --- Odometry ---
    int16_t delta_left  = current_ticks_left  - prev_ticks_left;
    int16_t delta_right = current_ticks_right - prev_ticks_right;

    float delta_s_left  = (float)delta_left  * DISTANCE_PER_TICK;
    float delta_s_right = (float)delta_right * DISTANCE_PER_TICK;
    float delta_s       = (delta_s_right + delta_s_left) / 2.0;
    float delta_theta   = (delta_s_right - delta_s_left) / TRACK_WIDTH_MM;

    total_distance += delta_s;
    robot_x        += delta_s * cos(current_angle);
    robot_y        += delta_s * sin(current_angle);
    current_angle  += delta_theta;

    prev_ticks_left  = current_ticks_left;
    prev_ticks_right = current_ticks_right;

    // --- Gyro Integration ---
    int16_t ax, ay, az, gx, gy, gz;

    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    float dt = (current_time - last_gyro_time) / 1000.0;
    last_gyro_time = current_time;

    float gz_cal = gz - gyro_offset_z;

    // Deadband to reduce noise
    if (abs(gz_cal) < 50)
        gz_cal = 0;

    // MPU6050 default sensitivity = 131 LSB/(deg/s)
    float gyro_rate_z = gz_cal / 131.0;

    heading_deg += gyro_rate_z * dt;

    /*Serial.print("Encoder Angle: ");
    Serial.print(current_angle * (180.0 / PI));

    Serial.print("  Gyro Angle: ");
    Serial.println(heading_deg);*/

        // --- P Controller (only during straight movement) ---
    if (pid_active) {
      pid_startup_count++;
      if (pid_startup_count <= PID_STARTUP_IGNORE) {
        ledcWrite(leftChannel,  BASE_SPEED);
        ledcWrite(rightChannel, BASE_SPEED);
      }
      else{
        float error = (float)(delta_right - delta_left);
        if (error < 0){
            error = -error;
        }
        integral += error;
        integral = constrain(integral, 0, 20);  // clamp it
        float correction = (Kp * error) + (Ki * integral);
        int left_pwm  = constrain((int)(BASE_SPEED + correction), 0, 255);
        int right_pwm = constrain((int)(BASE_SPEED - correction), 0, 255);
        Serial.printf("delta_left: %d, delta_right: %d, error: %.2f, integral: %.2f, left_pwm: %d, right_pwm: %d\n",
        delta_left, delta_right, error, integral, left_pwm, right_pwm);
        ledcWrite(leftChannel,  left_pwm);
        ledcWrite(rightChannel, right_pwm);
      }
    }

    // --- Broadcast telemetry ---
    char buf[80];
    snprintf(buf, sizeof(buf),
      "{\"x\":%.2f,\"y\":%.2f,\"theta\":%.2f,\"dist\":%.2f}",
      robot_x, robot_y, current_angle * (180.0 / PI), total_distance);
    webSocket.broadcastTXT(buf);
  }
}