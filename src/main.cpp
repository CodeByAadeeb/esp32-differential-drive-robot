#include "driver/pcnt.h" // is this a local library as it has ""
#include <Wire.h>
#include <MPU6050.h>
#include <WiFi.h>
#include <SPIFFS.h>
#include <ESPAsyncWebServer.h>
#include <WebSocketsServer.h>
#include <ArduinoOTA.h>
#include "my_functions.h"



AsyncWebServer   server(80);
WebSocketsServer webSocket = WebSocketsServer(1337);
//char msg_buf[10];



/***********************************************************
 * setup
 */
void setup() {
  Serial.begin(115200);
  Serial.printf("TICKS_PER_REVOLUTION: %.2f\n", TICKS_PER_REVOLUTION);
  Serial.printf("DISTANCE_PER_TICK: %.4f mm\n", DISTANCE_PER_TICK);

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
  ArduinoOTA.begin();
  Serial.println("\nAP running");
  Serial.print("My IP address: ");
  Serial.println(WiFi.softAPIP());

  server.on("/",          HTTP_GET, onIndexRequest);
  server.on("/style.css", HTTP_GET, onCSSRequest);
  server.onNotFound(onPageNotFound);
  server.begin();

  webSocket.begin();
  webSocket.enableHeartbeat(15000, 3000, 2);
  webSocket.onEvent(onWebSocketEvent);


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
  ArduinoOTA.handle();
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

    Serial.printf("ticks L:%d R:%d | delta L:%d R:%d\n", 
    current_ticks_left, current_ticks_right, delta_left, delta_right);

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
    float correction = 0.0, error = 0.0;
    int left_pwm = BASE_SPEED, right_pwm = BASE_SPEED;

        // --- P Controller (only during straight movement) ---
    if (pid_active) {
      pid_startup_count++;
      if (pid_startup_count <= PID_STARTUP_IGNORE) {
        ledcWrite(leftChannel,  BASE_SPEED);
        ledcWrite(rightChannel, BASE_SPEED);
      }
      else{
        error = (float)(delta_right - delta_left);
        integral += error;
        integral = constrain(integral, -50, 50);  // clamp it
        correction = (Kp * error) + (Ki * integral);
        left_pwm  = constrain((int)(BASE_SPEED + correction), 0, 255);
        right_pwm = constrain((int)(BASE_SPEED - correction), 0, 255);

        //Serial.printf("delta_left: %d, delta_right: %d, error: %.2f, integral: %.2f, left_pwm: %d, right_pwm: %d, current_ticks_left: %d, current_ticks_right: %d,",
        //    delta_left, delta_right, error, integral, left_pwm, right_pwm, current_ticks_left, current_ticks_right);
        ledcWrite(leftChannel,  left_pwm);
        ledcWrite(rightChannel, right_pwm);
      }
    }

    // After computing delta_left, delta_right and using them:
    if (abs(current_ticks_left) > 20000 || abs(current_ticks_right) > 20000) {
        pcnt_counter_clear(PCNT_UNIT_0);
        pcnt_counter_clear(PCNT_UNIT_1);
        prev_ticks_left = 0;
        prev_ticks_right = 0;
    }

    // --- Broadcast telemetry ---
    char buf[300];
    snprintf(buf, sizeof(buf),
      "{\"x\":%.2f,\"y\":%.2f,\"theta\":%.2f,\"dist\":%.2f,\"debug\":true,\"kp\":%.1f,\"error\":%.2f,\"integral\":%.2f,\"lpwm\":%d,\"rpwm\":%d,\"delta_right\":%d,\"delta_left\":%d}",
      robot_x, robot_y, current_angle * (180.0 / PI), total_distance, Kp, error, integral, left_pwm, right_pwm, delta_right, delta_left);
      Serial.println(buf);
    webSocket.broadcastTXT(buf);
  }
}