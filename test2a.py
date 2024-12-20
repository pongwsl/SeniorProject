# test2a.py
# created by pongwsl on 20 dec 2024
# lasted updated 20 dec 2024
# update from test1.py, to optimize the MediaPipe.
# This code tested on python 3.9.6 / 3.12.8
# To select Python Interpreter in VSCode:
# press cmd+shift+P, type 'Python: Select Interpreter'
# use "/Library/Developer/CommandLineTools/usr/bin/python3"

import cv2
import mediapipe as mp
import time
import threading

# Configuration Parameters
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_SKIP = 0  # Set to >0 to skip frames
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
MODEL_COMPLEXITY = 0  # 0 for lightweight, 1 for full

class VideoStream:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.ret, self.frame = self.capture.read()
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.capture.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.ret else (False, None)

    def stop(self):
        self.running = False
        self.thread.join()
        self.capture.release()

def main():
    # Initialize VideoStream
    video_stream = VideoStream(0)

    # Initialize MediaPipe Hands
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        model_complexity=MODEL_COMPLEXITY
    )
    mpDrawing = mp.solutions.drawing_utils

    # Variables for FPS calculation
    prev_time = time.time()
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        success, frame = video_stream.read()
        if not success or frame is None:
            break

        # Optional: Frame Skipping
        # Implement frame skipping logic here if FRAME_SKIP > 0

        # Preprocessing
        frame = cv2.flip(frame, 1)  # Flip horizontally
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and find hands using MediaPipe
        results = hands.process(rgb_frame)

        # Post-processing
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        # Overlay FPS on the frame
        cv2.putText(
            frame, f'FPS: {fps:.2f}', (10, 30),
            font, 1, (0, 255, 0), 2, cv2.LINE_AA
        )

        # Draw Hand Landmarks
        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks:
                mpDrawing.draw_landmarks(
                    frame, handLandmarks, mpHands.HAND_CONNECTIONS,
                    mpDrawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mpDrawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

        # Display the resulting frame
        cv2.imshow('Gesture Recognition', frame)

        # Exit Mechanism: Press any key to quit
        if cv2.waitKey(1) != -1: break

    # Cleanup
    video_stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()