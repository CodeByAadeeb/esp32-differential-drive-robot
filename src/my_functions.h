    #ifndef MY_FUNCTIONS_H
    #define MY_FUNCTIONS_H
    #include <Arduino.h>
    #include "driver/pcnt.h"
    #include <Wire.h>
    #include <MPU6050.h>
    #include <WiFi.h>
    #include <SPIFFS.h>
    #include <ESPAsyncWebServer.h>
    #include <WebSocketsServer.h>
    #include <ESP32Servo.h>
    #include <cmath>
    #include <WebSocketsClient.h> 

    // --- Hardware Configuration ---
    #define LEFT_ENCODER_A_PIN  4
    #define LEFT_ENCODER_B_PIN  32
    #define RIGHT_ENCODER_A_PIN 19
    #define RIGHT_ENCODER_B_PIN 18
    #define ENA 16
    #define IN1 14
    #define IN2 27
    #define ENB 26
    #define IN3 25
    #define IN4 33
    #define frequency   5000
    #define resolution  8
    #define leftChannel  2
    #define rightChannel 3
    #define SERVO_PIN 13
    #define ServoArraySize 19

    // --- Robot Constants ---
    extern const float WHEEL_DIAMETER_MM;
    //const float GEAR_RATIO        = 30.0;
    //const float ENCODER_BASE_PPR  = (693 + 696)/2;
    extern const float TRACK_WIDTH_MM;

    // --- Calculated Constants ---
    extern const float TICKS_PER_REVOLUTION;
    extern const float DISTANCE_PER_TICK;


    // --- Tracking Variables ---
    extern int16_t prev_ticks_left;
    extern int16_t prev_ticks_right;
    extern float total_distance;
    extern float current_angle;
    extern float robot_x;
    extern float robot_y;

    // --- Timing ---
    extern unsigned long last_loop_time;
    extern const unsigned long LOOP_INTERVAL_MS;

    // --- PID ---
    extern const float Kp;   // tune this: increase if still drifting, decrease if wobbling
    extern const int   BASE_SPEED;
    extern int16_t pid_ticks_left;   // tick snapshot when movement started
    extern int16_t pid_ticks_right;
    extern float integral;
    extern const float Ki;
    extern bool pid_active;
    extern int pid_startup_count;
    extern const int PID_STARTUP_IGNORE;  // ignore first 3 readings

    // --- Web Server & WebSocket ---
    extern const char *ssid;
    extern const char *password;
    extern const int http_port;
    extern const int ws_port;

    //AsyncWebServer   server(80);
    extern AsyncWebServer   server;
    extern WebSocketsServer webSocket;
    extern WebSocketsClient webSocketClient;  
    extern char msg_buf[10]; // wat is this for 

    // --- MPU6050 ---
    extern MPU6050 mpu;
    extern float gyro_offset_z;
    extern float heading_deg;
    extern unsigned long last_gyro_time;

    // --- Servo ---
    extern const int MIN_ANGLE;     // Semicircle start angle
    extern const int MAX_ANGLE;   // Semicircle end angle
    extern const int STEP_DELAY; // Semicircle end angle
    extern int current_servo_angle;
    extern bool sweepForward;
    extern float last_scan_x;
    extern float last_scan_y;
    extern const float MIN_RESCAN_DISTANCE;
    extern const int x_offset;
    extern const int y_offset;
    extern const int SERVO_STEP_SIZE;
    extern const unsigned long SERVO_SETTLE_MS;
    extern bool waiting_for_settle;
    extern unsigned long servo_step_time;

    // --- Sweep Scan Data Array ---
    const int SWEEP_SAMPLES = 181; // 0 to 180 degrees inclusive
    extern uint16_t sweep_distances[SWEEP_SAMPLES];
    extern bool is_sweeping;
    extern bool perm_to_scan;
    extern uint16_t scan_array[ServoArraySize];

    void setMotors(int left_speed, int right_speed);
    void onWebSocketEvent(uint8_t client_num, WStype_t type, uint8_t *payload, size_t length);
    void onPythonEvent(WStype_t type, uint8_t * payload, size_t length);
    void onIndexRequest(AsyncWebServerRequest *request);
    void onCSSRequest(AsyncWebServerRequest *request);
    void onPageNotFound(AsyncWebServerRequest *request);
    void init_hardware_pcnt(pcnt_unit_t unit, int gpio_a, int gpio_b);
    void calibrateMPU();



    #endif // MY_FUNCTIONS_H