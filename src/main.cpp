#include "driver/pcnt.h" // is this a local library as it has ""
#include <ArduinoOTA.h>
#include "my_functions.h"
#include <Adafruit_VL53L0X.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

#define DEBUG_SERIAL 0

#if DEBUG_SERIAL
  #define DEBUG_PRINTF(...) Serial.printf(__VA_ARGS__)
#else
  #define DEBUG_PRINTF(...) ((void)0)
#endif

AsyncWebServer   server(80);
WebSocketsClient webSocketClient; 
Adafruit_VL53L0X lox = Adafruit_VL53L0X();
Servo scannerServo;
//char msg_buf[10];

static void sendScanSample(int servo_angle, float x, float y, float theta, uint16_t* scan_data, int array_size) {
  if (!webSocketClient.isConnected()) {
    return;
  }

  // Large character buffer to comfortably fit 181 numbers plus metadata (~1000 bytes)
  char buf[1200]; 
  
  // Start formatting the beginning of your JSON string
  int len = snprintf(buf, sizeof(buf),
      "{\"type\":\"SCAN\",\"current_servo_angle\":%d,\"x\":%.2f,\"y\":%.2f,\"theta\":%.2f,\"measure\":[",
      servo_angle, x, y, theta);

  // 2. Loop through the array to add each element into the JSON array string
  for (int i = 0; i < array_size; i++) {
    // Append the number. If it is not the last item, append a comma
    int written = snprintf(buf + len, sizeof(buf) - len, "%u%s", 
                           scan_data[i], (i == array_size - 1) ? "" : ",");
    len += written;
    
    // Safety check to make sure we don't overflow our buffer
    if (len >= (int)sizeof(buf) - 5) break; 
  }

  // 3. Close the JSON array brackets and the root object
  int closing_len = snprintf(buf + len, sizeof(buf) - len, "]}");
  len += closing_len;

  // 4. Send the fully constructed JSON payload to Python
  if (len > 0 && len < (int)sizeof(buf)) {
    webSocketClient.sendTXT(buf);
  }
}

/***********************************************************
 * setup
 */
void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
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


  WiFi.begin("Redmi 12 5G", "aadeeb12"); // wifi to which laptop is connected to  ("Airtel_9945484809", "air79268")
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.print("\nConnected! ESP32 IP Address: ");
  Serial.println(WiFi.localIP());
  ArduinoOTA.begin();

  server.on("/",          HTTP_GET, onIndexRequest);
  server.on("/style.css", HTTP_GET, onCSSRequest);
  server.onNotFound(onPageNotFound);
  server.begin();

  webSocketClient.begin("10.24.150.23", 8765, "/"); // Python server IP  doubt on this line
  webSocketClient.onEvent(onPythonEvent);
  webSocketClient.enableHeartbeat(15000, 3000, 2);


  init_hardware_pcnt(PCNT_UNIT_0, LEFT_ENCODER_A_PIN,  LEFT_ENCODER_B_PIN);
  init_hardware_pcnt(PCNT_UNIT_1, RIGHT_ENCODER_A_PIN, RIGHT_ENCODER_B_PIN);

  Wire.begin(21, 23);

  mpu.initialize();

  calibrateMPU();

  if (!lox.begin()) {
    Serial.println("VL53L0X failed to start!");
    while (1);
  }
  
  lox.startRangeContinuous();  // start continuous mode

  last_gyro_time = millis();

  Serial.println("Hardware PCNT Odometry Initialized.");

  ESP32PWM::allocateTimer(3);
  scannerServo.setPeriodHertz(50);          // Standard 50Hz Servo Frequency
  int ch = scannerServo.attach(SERVO_PIN, 500, 2400);
  Serial.printf("Servo attach() returned channel: %d\n", ch);

  // Move to starting position smoothly
  scannerServo.write(MIN_ANGLE);
  //delay(1000);
}

/***********************************************************
 * loop
 */
void loop() {
  ArduinoOTA.handle();
  webSocketClient.loop();

  unsigned long current_time = millis();
  if (current_time - last_loop_time >= LOOP_INTERVAL_MS) {
    last_loop_time = current_time;

    // Read ticks
    int16_t current_ticks_left  = 0;
    int16_t current_ticks_right = 0;
    pcnt_get_counter_value(PCNT_UNIT_0, &current_ticks_left);
    pcnt_get_counter_value(PCNT_UNIT_1, &current_ticks_right);

    // --- Local odometry (ESP-side only) ---
    // NOTE: robot_x/robot_y/current_angle are NOT sent to Python as authoritative
    // pose anymore. Python's OdometryProvider recomputes x/y/theta itself from the
    // raw ticks_left/ticks_right/gyro_z_dps this loop broadcasts below — that's the
    // single source of truth now. These local values are kept only so the servo
    // scan math (x_scan/y_scan, SCAN message pose) has *something* to project
    // against on this side; they are sent to Python as esp_x/esp_y/esp_theta,
    // clearly separate from the corrected x/y/theta the browser ultimately sees.
    int16_t delta_left  = current_ticks_left  - prev_ticks_left;
    int16_t delta_right = current_ticks_right - prev_ticks_right;

    DEBUG_PRINTF("ticks L:%d R:%d | delta L:%d R:%d\n",
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

    // --- Gyro (raw rate only — no on-device integration sent downstream) ---
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    float dt = (current_time - last_gyro_time) / 1000.0;
    last_gyro_time = current_time;

    float gz_cal = gz - gyro_offset_z;

    // Deadband to reduce noise
    if (abs(gz_cal) < 50)
        gz_cal = 0;

    // MPU6050 default sensitivity = 131 LSB/(deg/s)
    float gyro_rate_z = gz_cal / 131.0;   // deg/s — this raw rate is what gets sent to Python
    heading_deg += gyro_rate_z * dt;      // kept locally only, for your own Serial debugging if needed

    VL53L0X_RangingMeasurementData_t measure;
    uint16_t distance_mm = 8190;  // default = out of range
    if (lox.isRangeComplete()) {
      distance_mm = lox.readRangeResult();
    }

    DEBUG_PRINTF("perm_to_scan = %d\n", perm_to_scan);

    // rotate the servo for scanning
    if (abs(delta_s) < 0.5 && perm_to_scan) {
      if (!waiting_for_settle) {
        // Just arrived at this angle (or first entry) — start the settle timer, don't sample yet
        waiting_for_settle = true;
        servo_step_time = millis();
      }
      else if (millis() - servo_step_time >= SERVO_SETTLE_MS) {
        // Servo has had time to stop vibrating — safe to trust this reading
        if (lox.isRangeComplete()) {
          distance_mm = lox.readRangeResult();
        }
        int scan_index = current_servo_angle / SERVO_STEP_SIZE;
        if (scan_index < 0) scan_index = 0;
        if (scan_index >= ServoArraySize) scan_index = ServoArraySize - 1;

        DEBUG_PRINTF("WRITE scan_array[%d] = %u (angle=%d, rangeComplete=%d)\n",
                     scan_index, distance_mm, current_servo_angle, lox.isRangeComplete());

        scan_array[scan_index] = distance_mm;

        if (sweepForward) {
          current_servo_angle += SERVO_STEP_SIZE;
          if (current_servo_angle >= MAX_ANGLE) {
            webSocketClient.sendTXT("{\"action\":\"start\",\"target\":\"image_processing\",\"status\":\"initiated\"}");
            current_servo_angle = MAX_ANGLE;
            sweepForward = false;
            perm_to_scan = false; 
            sendScanSample(current_servo_angle, robot_x, robot_y, current_angle, scan_array, ServoArraySize);
          }
        } else {
          current_servo_angle -= SERVO_STEP_SIZE;
          if (current_servo_angle <= MIN_ANGLE) {
            webSocketClient.sendTXT("{\"action\":\"start\",\"target\":\"image_processing\",\"status\":\"initiated\"}");
            current_servo_angle = MIN_ANGLE;
            sweepForward = true;
            perm_to_scan = false; 
            sendScanSample(current_servo_angle, robot_x, robot_y, current_angle, scan_array, ServoArraySize);
          }
        }
        scannerServo.write(current_servo_angle);
        waiting_for_settle = false; // will re-arm the settle timer once we hit the next angle
      }
    }

    if (abs(delta_s) > 0.5){
      perm_to_scan = true;
    }

    // scan coordinates (still projected from ESP-local pose — see note above)
    float sensor_world_angle = current_angle - (PI / 2.0);
    float x_scan = (distance_mm * cos(sensor_world_angle)) + robot_x + x_offset;
    float y_scan = (distance_mm * sin(sensor_world_angle)) + robot_y + y_offset;

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

        // Flip correction direction for backward movement
        if (delta_left < 0 && delta_right < 0) {
            correction = -((Kp * error) + (Ki * integral));
        } else {
            correction = (Kp * error) + (Ki * integral);
        }
        left_pwm  = constrain((int)(BASE_SPEED + correction), 0, 255);
        right_pwm = constrain((int)(BASE_SPEED - correction), 0, 255);

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
    // Raw inputs (ticks_left, ticks_right, gyro_z_dps) are what Python's
    // OdometryProvider actually consumes as the single source of pose truth.
    // esp_x/esp_y/esp_theta are ESP-local dead reckoning only, kept for
    // reference/debugging — Python overwrites x/y/theta before it reaches
    // the browser, so these are intentionally namespaced to avoid confusion.
    char buf[480];
    snprintf(buf, sizeof(buf),
      "{\"ticks_left\":%d,\"ticks_right\":%d,\"gyro_z_dps\":%.3f,"
      "\"esp_x\":%.2f,\"esp_y\":%.2f,\"esp_theta\":%.2f,"
      "\"dist\":%.2f,\"debug\":true,\"kp\":%.1f,\"error\":%.2f,\"integral\":%.2f,"
      "\"lpwm\":%d,\"rpwm\":%d,\"delta_right\":%d,\"delta_left\":%d,"
      "\"measure\":%d,\"x_scan\":%.2f,\"y_scan\":%.2f,\"delta_s\":%.2f}",
      current_ticks_left, current_ticks_right, gyro_rate_z,
      robot_x, robot_y, current_angle * (180.0 / PI),
      total_distance, Kp, error, integral, left_pwm, right_pwm,
      delta_right, delta_left, distance_mm, x_scan, y_scan, delta_s);
    DEBUG_PRINTF("%s\n", buf);
    webSocketClient.sendTXT(buf);

  }
}