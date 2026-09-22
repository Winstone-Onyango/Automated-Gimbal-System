import cv2
import numpy as np
import urllib.request
import os
import time
import threading

# Raspberry Pi GPIO control
try:
    import RPi.GPIO as GPIO

    RPI_MODE = True
    print("✓ Running on Raspberry Pi")
except ImportError:
    RPI_MODE = False
    print("⚠ Not running on Raspberry Pi - SIMULATION MODE")

# ===== CONFIGURATION =====

# URL of the IP camera stream
# camera_url = "http://192.168.100.132:8080/video"
camera_url_two = "http://10.179.72.36:8080/video"
# ===== STEPPER MOTOR GPIO PINS (BCM NUMBERING) =====
# Using A4988 or DRV8825 stepper drivers

# Pan Motor (Horizontal/X-axis)
PAN_STEP_PIN = 17  # GPIO 17 (Physical pin 11)
PAN_DIR_PIN = 27  # GPIO 27 (Physical pin 13)
PAN_ENABLE_PIN = 22  # GPIO 22 (Physical pin 15) - Optional

# Tilt Motor (Vertical/Y-axis)
TILT_STEP_PIN = 23  # GPIO 23 (Physical pin 16)
TILT_DIR_PIN = 24  # GPIO 24 (Physical pin 18)
TILT_ENABLE_PIN = 25  # GPIO 25 (Physical pin 22) - Optional

# ===== STEPPER MOTOR SPECIFICATIONS =====
STEPS_PER_REVOLUTION = 200  # NEMA 17: 1.8° per step
MICROSTEPS = 8  # OPTIMIZED: Changed from 16 to 8 for more torque at speed (Set on driver: MS1, MS2, MS3)
TOTAL_STEPS = STEPS_PER_REVOLUTION * MICROSTEPS  # 1600 steps/revolution

# ===== SPEED SETTINGS (OPTIMIZED FOR SMOOTH, FAST MOVEMENT) =====
STEP_DELAY = 0.00012  # Base step period for small moves
MAX_SPEED_DELAY = 0.00008  # Fastest step period for large corrections
MIN_SPEED_DELAY = 0.00035  # Slowest step period for small corrections
PULSE_HIGH_TIME = 0.00002  # High pulse width for stepper driver

# ===== ACCELERATION RAMPING =====
ENABLE_ACCELERATION = True  # Enable smooth acceleration/deceleration
ACCEL_FACTOR = 0.94  # Speed up by 6% each step (0.94 = multiply delay by 0.94)
DECEL_FACTOR = 1.06  # Slow down by 6% each step

# ===== TRACKING PARAMETERS =====
DEAD_ZONE = 30  # Pixels - ignore movements smaller than this
SMOOTHING_FACTOR = 0.45  # 0.0-1.0 (lower = smoother, higher = more responsive)
STEPS_PER_ADJUSTMENT = 10  # Base step count per adjustment
MAX_ADJUSTMENT_STEPS = 40  # Limit step size to prevent abrupt jumps

# ===== DNN MODEL FILES =====
model_file = "res10_300x300_ssd_iter_140000.caffemodel"
config_file = "deploy.prototxt"
model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
config_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"

# ===== HELPER FUNCTIONS =====


def download_file(url, filename):
    """Download file if it doesn't exist"""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"✓ {filename} downloaded successfully!")
        except Exception as e:
            print(f"✗ Error downloading {filename}: {e}")
            return False
    else:
        print(f"✓ {filename} already exists")
    return True


# ===== STEPPER MOTOR CLASS =====


class StepperMotor:
    """Control a stepper motor using A4988/DRV8825 driver on Raspberry Pi GPIO"""

    def __init__(self, step_pin, dir_pin, enable_pin=None, name="Motor"):
        self.step_pin = step_pin
        self.dir_pin = dir_pin
        self.enable_pin = enable_pin
        self.name = name
        self.current_position = 0  # Track position in steps
        self.is_moving = False
        self.lock = threading.Lock()

        if RPI_MODE:
            # Setup GPIO pins
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.step_pin, GPIO.OUT)
            GPIO.setup(self.dir_pin, GPIO.OUT)

            if self.enable_pin:
                GPIO.setup(self.enable_pin, GPIO.OUT)
                GPIO.output(self.enable_pin, GPIO.LOW)  # LOW = enabled

            # Initialize to LOW
            GPIO.output(self.step_pin, GPIO.LOW)
            GPIO.output(self.dir_pin, GPIO.LOW)

            print(f"✓ {name} initialized on GPIO pins:")
            print(f"    STEP: {step_pin}, DIR: {dir_pin}, EN: {enable_pin}")

    def enable(self):
        """Enable the motor (allows movement, consumes power)"""
        if RPI_MODE and self.enable_pin:
            GPIO.output(self.enable_pin, GPIO.LOW)  # LOW = enabled
            print(f"{self.name} enabled")

    def disable(self):
        """Disable the motor (saves power, reduces heat)"""
        if RPI_MODE and self.enable_pin:
            GPIO.output(self.enable_pin, GPIO.HIGH)  # HIGH = disabled
            print(f"{self.name} disabled")

    def step(self, steps, direction=1, delay=STEP_DELAY):
        """
        Move motor by specified number of steps with acceleration ramping

        Args:
            steps: Number of steps to move
            direction: 1 = clockwise, -1 = counter-clockwise
            delay: Delay between steps in seconds (controls speed)
        """
        if steps == 0:
            return

        if not RPI_MODE:
            # Simulation mode
            self.current_position += steps * direction
            print(
                f"{self.name}: {steps} steps {'CW' if direction > 0 else 'CCW'}, "
                f"Position: {self.current_position}"
            )
            return

        with self.lock:
            # Set direction
            if direction > 0:
                GPIO.output(self.dir_pin, GPIO.HIGH)  # Clockwise
            else:
                GPIO.output(self.dir_pin, GPIO.LOW)  # Counter-clockwise

            time.sleep(0.000008)  # OPTIMIZED: Direction setup time increased to 8μs (was 1μs)

            if ENABLE_ACCELERATION and steps > 5:
                # Acceleration ramping for smooth movement
                accel_steps = min(steps // 3, 50)  # Ramp up/down over ~33% of movement
                decel_start = steps - accel_steps
                current_delay = 0.0015  # Start slow and ramp into the target delay

                for i in range(abs(steps)):
                    # Acceleration phase
                    if i < accel_steps:
                        current_delay *= ACCEL_FACTOR  # Speed up
                        current_delay = max(current_delay, delay)
                    # Deceleration phase
                    elif i >= decel_start:
                        current_delay *= DECEL_FACTOR  # Slow down

                    # Generate step pulse
                    GPIO.output(self.step_pin, GPIO.HIGH)
                    pulse_time = min(PULSE_HIGH_TIME, current_delay / 2)
                    time.sleep(pulse_time)
                    GPIO.output(self.step_pin, GPIO.LOW)
                    time.sleep(max(0.0, current_delay - pulse_time))
            else:
                # Simple stepping without acceleration for small movements
                for _ in range(abs(steps)):
                    GPIO.output(self.step_pin, GPIO.HIGH)
                    pulse_time = min(PULSE_HIGH_TIME, delay / 2)
                    time.sleep(pulse_time)
                    GPIO.output(self.step_pin, GPIO.LOW)
                    time.sleep(max(0.0, delay - pulse_time))

            # Update position
            self.current_position += steps * direction

    def step_async(self, steps, direction=1, delay=STEP_DELAY):
        """Move motor asynchronously in a separate thread"""
        if self.is_moving:
            return  # Skip if already moving

        def move():
            self.is_moving = True
            self.step(steps, direction, delay)
            self.is_moving = False

        thread = threading.Thread(target=move, daemon=True)
        thread.start()

    def goto_position(self, target_position, delay=STEP_DELAY):
        """Move to absolute position"""
        steps_to_move = target_position - self.current_position
        if steps_to_move == 0:
            return

        direction = 1 if steps_to_move > 0 else -1
        self.step(abs(steps_to_move), direction, delay)

    def reset_position(self):
        """Reset position counter to zero (home position)"""
        self.current_position = 0
        print(f"{self.name} position reset to 0")


# ===== FACE TRACKING CONTROLLER =====


class FaceTrackingController:
    """Manages pan/tilt stepper motors for face tracking"""

    def __init__(self):
        print("\n" + "=" * 50)
        print("Initializing Face Tracking Controller")
        print("=" * 50)

        # Initialize motors
        self.pan_motor = StepperMotor(
            PAN_STEP_PIN, PAN_DIR_PIN, PAN_ENABLE_PIN, "Pan Motor"
        )
        self.tilt_motor = StepperMotor(
            TILT_STEP_PIN, TILT_DIR_PIN, TILT_ENABLE_PIN, "Tilt Motor"
        )

        # Enable motors
        self.pan_motor.enable()
        self.tilt_motor.enable()

        print("✓ Face Tracking Controller ready")
        print("=" * 50 + "\n")

    def adjust_pan(self, offset_x, frame_width):
        """
        Adjust pan motor based on horizontal offset

        Args:
            offset_x: Horizontal offset from center (positive = face is right)
            frame_width: Width of frame in pixels
        """
        if abs(offset_x) < DEAD_ZONE:
            return  # Within dead zone, don't move

        # Calculate steps proportional to offset
        # Larger offset = more steps, but keep motion smooth
        steps = int((abs(offset_x) / frame_width) * STEPS_PER_ADJUSTMENT * 3)
        steps = max(1, min(steps, MAX_ADJUSTMENT_STEPS))

        # Direction: positive offset means face is to the right
        # Motor should turn right (clockwise) to center it
        direction = 1 if offset_x > 0 else -1

        # Calculate speed based on offset magnitude
        speed_factor = min(abs(offset_x) / (frame_width / 2), 1.0)
        delay = MIN_SPEED_DELAY + (MAX_SPEED_DELAY - MIN_SPEED_DELAY) * (
            1 - speed_factor
        )

        self.pan_motor.step_async(steps, direction, delay)

    def adjust_tilt(self, offset_y, frame_height):
        """
        Adjust tilt motor based on vertical offset

        Args:
            offset_y: Vertical offset from center (positive = face is down)
            frame_height: Height of frame in pixels
        """
        if abs(offset_y) < DEAD_ZONE:
            return  # Within dead zone, don't move

        # Calculate steps
        steps = int((abs(offset_y) / frame_height) * STEPS_PER_ADJUSTMENT * 3)
        steps = max(1, min(steps, MAX_ADJUSTMENT_STEPS))

        # Direction: positive offset means face is below center
        # Motor should tilt down to center it
        direction = 1 if offset_y > 0 else -1

        # Calculate speed
        speed_factor = min(abs(offset_y) / (frame_height / 2), 1.0)
        delay = MIN_SPEED_DELAY + (MAX_SPEED_DELAY - MIN_SPEED_DELAY) * (
            1 - speed_factor
        )

        self.tilt_motor.step_async(steps, direction, delay)

    def center(self):
        """Return both motors to center position (0, 0)"""
        print("\n" + "=" * 40)
        print("Centering motors...")
        print("=" * 40)
        self.pan_motor.goto_position(0)
        self.tilt_motor.goto_position(0)
        print("✓ Motors centered at position (0, 0)")
        print("=" * 40 + "\n")

    def get_status(self):
        """Get current motor positions"""
        return {
            "pan": self.pan_motor.current_position,
            "tilt": self.tilt_motor.current_position,
            "pan_moving": self.pan_motor.is_moving,
            "tilt_moving": self.tilt_motor.is_moving,
        }

    def disable_motors(self):
        """Disable both motors (save power)"""
        self.pan_motor.disable()
        self.tilt_motor.disable()

    def enable_motors(self):
        """Enable both motors"""
        self.pan_motor.enable()
        self.tilt_motor.enable()

    def cleanup(self):
        """Clean up GPIO resources"""
        print("\n" + "=" * 40)
        print("Cleaning up...")
        print("=" * 40)
        self.disable_motors()
        if RPI_MODE:
            GPIO.cleanup()
            print("✓ GPIO cleaned up")
        print("=" * 40 + "\n")


# ===== MAIN PROGRAM =====


def main():
    print("\n" + "=" * 60)
    print("  FACE TRACKING SYSTEM - RASPBERRY PI + STEPPER MOTORS")
    print("  [OPTIMIZED VERSION - SMOOTH & FAST MOVEMENT]")
    print("=" * 60 + "\n")

    # Download and load DNN model
    print("Step 1: Checking DNN model files...")
    if not download_file(config_url, config_file):
        return
    if not download_file(model_url, model_file):
        return

    print("\nStep 2: Loading face detection model...")
    try:
        face_net = cv2.dnn.readNetFromCaffe(config_file, model_file)
        print("✓ DNN model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    # Initialize face tracking controller
    print("\nStep 3: Initializing stepper motors...")
    controller = FaceTrackingController()
    time.sleep(1)  # Let motors settle

    # Initialize video capture
    print("Step 4: Connecting to IP camera...")
    cap = cv2.VideoCapture(camera_url_two)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"✗ Error: Could not connect to {camera_url_two}")
        controller.cleanup()
        return

    print(f"✓ Connected to camera: {camera_url_two}")

    # Display controls
    print("\n" + "=" * 60)
    print("CONTROLS:")
    print("  q - Quit program")
    print("  c - Center motors (return to home position)")
    print("  d - Disable motors (save power)")
    print("  e - Enable motors")
    print("  r - Reset position counters to zero")
    print("=" * 60 + "\n")

    print("Starting face tracking...\n")

    # Tracking variables
    frame_count = 0
    skip_frames = 1  # Process every 2nd frame for better responsiveness
    confidence_threshold = 0.6
    resize_width = 320  # Lower image size for faster detection
    smoothed_offset_x = 0
    smoothed_offset_y = 0
    last_face_center = None  # Keep previous face location for stable tracking

    # FPS calculation
    fps_start_time = time.time()
    fps_counter = 0
    current_fps = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("✗ Error: Failed to grab frame")
                break

            frame_count += 1
            fps_counter += 1

            # Calculate FPS
            if time.time() - fps_start_time >= 1.0:
                current_fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()

            original_h, original_w = frame.shape[:2]

            # Calculate frame center
            center_x = original_w // 2
            center_y = original_h // 2

            # Draw crosshair at center
            cv2.line(
                frame,
                (center_x - 20, center_y),
                (center_x + 20, center_y),
                (0, 255, 255),
                2,
            )
            cv2.line(
                frame,
                (center_x, center_y - 20),
                (center_x, center_y + 20),
                (0, 255, 255),
                2,
            )
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)

            # Process keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\nQuitting...")
                break
            elif key == ord("c"):
                controller.center()
            elif key == ord("d"):
                controller.disable_motors()
            elif key == ord("e"):
                controller.enable_motors()
            elif key == ord("r"):
                controller.pan_motor.reset_position()
                controller.tilt_motor.reset_position()

            # Skip frames for performance
            if frame_count % (skip_frames + 1) != 0:
                # cv2.imshow("Face Tracking - Raspberry Pi", frame)
                continue

            # Resize for faster processing
            aspect_ratio = original_h / original_w
            resize_height = int(resize_width * aspect_ratio)
            resized_frame = cv2.resize(frame, (resize_width, resize_height))
            h, w = resized_frame.shape[:2]

            # Detect faces
            blob = cv2.dnn.blobFromImage(
                resized_frame, 1.0, (300, 300), (104.0, 177.0, 123.0)
            )
            face_net.setInput(blob)

            try:
                detections = face_net.forward()
            except Exception as e:
                print(f"Detection error: {e}")
                continue

            # Find best face: prefer the one closest to the previous detection
            best_face = None
            best_confidence = 0
            best_near_face = None
            best_near_confidence = 0
            closest_distance = float("inf")

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence <= confidence_threshold:
                    continue

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")

                # Scale to original size
                scale_x = original_w / w
                scale_y = original_h / h

                x = int(max(0, x * scale_x))
                y = int(max(0, y * scale_y))
                x2 = int(min(original_w, x2 * scale_x))
                y2 = int(min(original_h, y2 * scale_y))

                face_center_x = x + (x2 - x) // 2
                face_center_y = y + (y2 - y) // 2

                if confidence > best_confidence:
                    best_face = (x, y, x2, y2)
                    best_confidence = confidence

                if last_face_center is not None:
                    dx = face_center_x - last_face_center[0]
                    dy = face_center_y - last_face_center[1]
                    distance_sq = dx * dx + dy * dy

                    if distance_sq < closest_distance or (
                        distance_sq == closest_distance
                        and confidence > best_near_confidence
                    ):
                        closest_distance = distance_sq
                        best_near_face = (x, y, x2, y2)
                        best_near_confidence = confidence

            if best_near_face is not None:
                roi_radius = max(original_w, original_h) * 0.35
                if closest_distance <= roi_radius * roi_radius:
                    best_face = best_near_face
                    best_confidence = best_near_confidence

            # Process best face
            if best_face:
                x, y, x2, y2 = best_face
                face_width = x2 - x
                face_height = y2 - y

                # Calculate face center
                face_center_x = x + face_width // 2
                face_center_y = y + face_height // 2
                last_face_center = (face_center_x, face_center_y)

                # Calculate offset from frame center
                offset_x = face_center_x - center_x
                offset_y = face_center_y - center_y

                # Apply smoothing
                smoothed_offset_x = int(
                    smoothed_offset_x * (1 - SMOOTHING_FACTOR)
                    + offset_x * SMOOTHING_FACTOR
                )
                smoothed_offset_y = int(
                    smoothed_offset_y * (1 - SMOOTHING_FACTOR)
                    + offset_y * SMOOTHING_FACTOR
                )

                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (face_center_x, face_center_y), 8, (255, 0, 0), -1)
                cv2.line(
                    frame,
                    (center_x, center_y),
                    (face_center_x, face_center_y),
                    (255, 0, 255),
                    2,
                )

                # Get motor status
                status = controller.get_status()
                print(status)
                # Display info
                cv2.putText(
                    frame,
                    f"Offset X: {smoothed_offset_x}px",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Offset Y: {smoothed_offset_y}px",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Pan: {status['pan']} steps",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Tilt: {status['tilt']} steps",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"FPS: {current_fps:.1f}",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Confidence: {best_confidence:.1%}",
                    (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # Adjust motors if outside dead zone
                if (
                    abs(smoothed_offset_x) > DEAD_ZONE
                    or abs(smoothed_offset_y) > DEAD_ZONE
                ):
                    controller.adjust_pan(smoothed_offset_x, original_w)
                    controller.adjust_tilt(smoothed_offset_y, original_h)
                    cv2.putText(
                        frame,
                        "TRACKING",
                        (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        frame,
                        "CENTERED",
                        (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
            else:
                cv2.putText(
                    frame,
                    "NO FACE DETECTED",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"FPS: {current_fps:.1f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

            # cv2.imshow("Face Tracking - Raspberry Pi", frame)

    except KeyboardInterrupt:
        print("\n✗ Interrupted by user (Ctrl+C)")

    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        print("\nShutting down...")
        controller.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        print("✓ System shutdown complete\n")


# ===== ENTRY POINT =====

if __name__ == "__main__":
    main()