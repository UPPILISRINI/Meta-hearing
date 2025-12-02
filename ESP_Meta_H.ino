#include <WiFi.h>
#include <FirebaseESP32.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// Wi-Fi Credentials
#define WIFI_SSID "Uppilis"
#define WIFI_PASSWORD "uppili2442"

// Firebase Credentials
#define API_KEY "AIzaSyCXeTza_jl9A95Thgb4EVbzcdxSMMIk8wM"
#define DATABASE_URL "https://project-259c7-default-rtdb.firebaseio.com/"

// Firebase Authentication
#define USER_EMAIL "project@gmail.com"
#define USER_PASSWORD "123456789"

// OLED Display Setup
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 32
#define OLED_RESET -1
#define SCREEN_ADDRESS 0x3C

// Define SDA and SCL pins for ESP32
#define SDA_PIN 21
#define SCL_PIN 22


void displayScrollingMessage(String msg) {
    display.clearDisplay();
    lineHeight = 10; // Height of each text line
    int numLines = SCREEN_HEIGHT / lineHeight; // Max lines that fit on screen
    int totalLines = (msg.length() / 16) + 1; // Estimate total lines based on width

    for (int i = 0; i < totalLines; i++) {
        display.clearDisplay();
        display.setCursor(0, 0);

        // Print multiple lines within the screen height
        for (int j = 0; j < numLines; j++) {
            int index = (i + j) * 25; // 16 characters per line (approx.)
            if (index < msg.length()) {
                display.println(msg.substring(index, index + 25));
            }
        }

        display.display();
        delay(500); // Adjust speed of scrolling
    }
}

void displayMessage(String msg) {
    display.clearDisplay();
    display.setCursor(0, 10);
    display.println(msg);
    display.display();
}
