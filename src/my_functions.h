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
//const float GEAR_RATIO        = 30.0;
//const float ENCODER_BASE_PPR  = (693 + 696)/2;
const float TRACK_WIDTH_MM    = 122.0;

// --- Calculated Constants ---
const float TICKS_PER_REVOLUTION  = (693 + 696);
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
const float Kp          = 2.1;   // tune this: increase if still drifting, decrease if wobbling
const int   BASE_SPEED  = 180;
int16_t pid_ticks_left  = 0;   // tick snapshot when movement started
int16_t pid_ticks_right = 0;
float integral = 0.0;
const float Ki = 0.8;
bool pid_active = false;
int pid_startup_count = 0;
const int PID_STARTUP_IGNORE = 3;  // ignore first 3 readings

// --- Web Server & WebSocket ---
const char *ssid     = "ESP32-AP";
const char *password = "LetMeInPlz";
const int http_port  = 80;
const int ws_port    = 1337;

//AsyncWebServer   server(80);
extern WebSocketsServer webSocket;
char msg_buf[10];

// --- MPU6050 ---
MPU6050 mpu;
float gyro_offset_z = 0;
float heading_deg = 0;
unsigned long last_gyro_time = 0;

void setMotors(int left_speed, int right_speed);
void onWebSocketEvent(uint8_t client_num, WStype_t type, uint8_t *payload, size_t length);
void onIndexRequest(AsyncWebServerRequest *request);
void onCSSRequest(AsyncWebServerRequest *request);
void onPageNotFound(AsyncWebServerRequest *request);
void init_hardware_pcnt(pcnt_unit_t unit, int gpio_a, int gpio_b);
void calibrateMPU();



#endif // MY_FUNCTIONS_H