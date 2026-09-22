import RPi.GPIO as GPIO
import time

# -----------------------------
# PIN SETUP
# -----------------------------
PAN_STEP_PIN = 17
PAN_DIR_PIN = 27
PAN_ENABLE_PIN = 22

TILT_STEP_PIN = 23
TILT_DIR_PIN = 24
TILT_ENABLE_PIN = 25

# -----------------------------
# CONFIG
# -----------------------------
STEPS = 5  # number of steps per move
DELAY = 0.001  # speed (lower = faster)

# -----------------------------
# GPIO SETUP
# -----------------------------
GPIO.setmode(GPIO.BCM)

pins = [
    PAN_STEP_PIN,
    PAN_DIR_PIN,
    PAN_ENABLE_PIN,
    TILT_STEP_PIN,
    TILT_DIR_PIN,
    TILT_ENABLE_PIN,
]

for pin in pins:
    GPIO.setup(pin, GPIO.OUT)

# Enable drivers (LOW = enabled for DRV8825)
GPIO.output(PAN_ENABLE_PIN, GPIO.LOW)
GPIO.output(TILT_ENABLE_PIN, GPIO.LOW)


# -----------------------------
# STEP FUNCTION
# -----------------------------
def step_motor(step_pin, dir_pin, direction, steps):
    GPIO.output(dir_pin, direction)

    for _ in range(steps):
        GPIO.output(step_pin, GPIO.HIGH)
        time.sleep(DELAY)
        GPIO.output(step_pin, GPIO.LOW)
        time.sleep(DELAY)


# -----------------------------
# TEST SEQUENCE
# -----------------------------
try:
    print("Pan right")
    step_motor(PAN_STEP_PIN, PAN_DIR_PIN, GPIO.HIGH, STEPS)
    time.sleep(1)

    print("Pan left")
    step_motor(PAN_STEP_PIN, PAN_DIR_PIN, GPIO.LOW, STEPS)
    time.sleep(1)

    print("Tilt up")
    step_motor(TILT_STEP_PIN, TILT_DIR_PIN, GPIO.HIGH, STEPS)
    time.sleep(1)

    print("Tilt down")
    step_motor(TILT_STEP_PIN, TILT_DIR_PIN, GPIO.LOW, STEPS)
    time.sleep(1)

    print("Combined movement")
    for _ in range(STEPS):
        GPIO.output(PAN_STEP_PIN, GPIO.HIGH)
        GPIO.output(TILT_STEP_PIN, GPIO.HIGH)
        time.sleep(DELAY)

        GPIO.output(PAN_STEP_PIN, GPIO.LOW)
        GPIO.output(TILT_STEP_PIN, GPIO.LOW)
        time.sleep(DELAY)

    print("Done!")

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    GPIO.cleanup()
