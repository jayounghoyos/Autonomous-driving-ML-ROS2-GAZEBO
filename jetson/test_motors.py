#!/usr/bin/env python3
"""
Test motor control on Jetson Nano with L298N.

Wiring:
  IN1 → Pin 11 (GPIO17) - Left motor direction 1
  IN2 → Pin 12 (GPIO18) - Left motor direction 2
  IN3 → Pin 13 (GPIO27) - Right motor direction 1
  IN4 → Pin 15 (GPIO22) - Right motor direction 2
  ENA/ENB → Jumpered (full speed)

Run on Jetson Nano:
  python3 test_motors.py
"""

import time

try:
    import Jetson.GPIO as GPIO
except ImportError:
    print("ERROR: Jetson.GPIO not found. Run this on Jetson Nano!")
    print("If on Jetson, install with: sudo pip3 install Jetson.GPIO")
    exit(1)

# Pin definitions (BOARD numbering)
IN1 = 11  # Left motor direction 1
IN2 = 12  # Left motor direction 2
IN3 = 13  # Right motor direction 1
IN4 = 15  # Right motor direction 2

def setup():
    """Initialize GPIO pins."""
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    GPIO.setup([IN1, IN2, IN3, IN4], GPIO.OUT)
    print("GPIO initialized")
    print(f"  Left motor:  IN1=Pin{IN1}, IN2=Pin{IN2}")
    print(f"  Right motor: IN3=Pin{IN3}, IN4=Pin{IN4}")

def stop():
    """Stop both motors."""
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    print("STOP")

def forward():
    """Both motors forward."""
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    print("FORWARD")

def backward():
    """Both motors backward."""
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    print("BACKWARD")

def turn_left():
    """Turn left (right forward, left backward)."""
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    print("TURN LEFT")

def turn_right():
    """Turn right (left forward, right backward)."""
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    print("TURN RIGHT")

def cleanup():
    """Clean up GPIO."""
    stop()
    GPIO.cleanup()
    print("GPIO cleaned up")

def main():
    print("=" * 50)
    print("L298N Motor Test - Jetson Nano")
    print("=" * 50)
    print("\nThis will test all motor directions.")
    print("Make sure robot is on a safe surface or elevated!\n")

    input("Press ENTER to start test (Ctrl+C to cancel)...")

    try:
        setup()

        print("\n--- Test 1: Forward (2 seconds) ---")
        forward()
        time.sleep(2)
        stop()
        time.sleep(1)

        print("\n--- Test 2: Backward (2 seconds) ---")
        backward()
        time.sleep(2)
        stop()
        time.sleep(1)

        print("\n--- Test 3: Turn Left (2 seconds) ---")
        turn_left()
        time.sleep(2)
        stop()
        time.sleep(1)

        print("\n--- Test 4: Turn Right (2 seconds) ---")
        turn_right()
        time.sleep(2)
        stop()
        time.sleep(1)

        print("\n" + "=" * 50)
        print("Test complete! All directions working?")
        print("=" * 50)

    except KeyboardInterrupt:
        print("\nTest cancelled")
    finally:
        cleanup()

if __name__ == "__main__":
    main()
