# test3.py
# created by pongwsl on 20 dec 2024
# lasted updated 20 dec 2024
# update from test2a.py, to add def tipCoord(lm, i).
# This code tested on python 3.9.13 on venv-metal.
# To select Python Interpreter in VSCode: press [[cmd + shift + P]], type 'Python: Select Interpreter'
# use "/Users/wasinlapthanaphat/Desktop/helloWorld/SeniorProject-backup/venv-metal/bin/python"

import cv2
import mediapipe as mp
import time
import threading
import math

# # Global variable for landmarks
# handLandmarks = None
# worldLandmarks = None

# Configuration Parameters
frameWidth = 640
frameHeight = 480
frameSkip = 0  # Set to >0 to skip frames
maxNumHands = 1
minDetectionConfidence = 0.8  # higher is faster (GPT's opinion)
minTrackingConfidence = 0.8
modelComplexity = 0  # 0 for lightweight, 1 for full

class VideoStream:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
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

# def link(n, m, a):
#     return handLandmarks.landmark[n].a - handLandmarks.landmark[m].a
def link(lm, n, m, a):
    return getattr(lm.landmark[n], a) - getattr(lm.landmark[m], a)

def length(lm, n, m):
    return math.sqrt(
        link(lm, n, m, 'x') ** 2 +
        link(lm, n, m, 'y') ** 2 +
        link(lm, n, m, 'z') ** 2
    )

def pointerData(lm):  # Nozzle on/off
    rollAngle = math.degrees(math.atan(link(lm, 5, 17, 'y') / link(lm, 5, 17, 'x')))
    pitchAngle = math.degrees(math.asin(link(lm, 8, 5, 'y') / length(lm, 8, 5)))
    yawAngle = math.degrees(math.asin(link(lm, 8, 5, 'x') / length(lm, 8, 5)))

    # Pinching Action
    if length(lm, 4, 3) <= length(lm, 8, 4): 
        pinchingAction = 'Nozzle off'
    else:
        pinchingAction = 'Nozzle on'
    
    return (pinchingAction, round(rollAngle, 2), round(pitchAngle, 2), round(yawAngle, 2))

def main():
    # # Declare as global to modify it within the function
    # global handLandmarks
    # global worldLandmarks

    # Initialize VideoStream
    videoStream = VideoStream(0)

    # Initialize MediaPipe Hands
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode        = False,
        max_num_hands            = maxNumHands,
        min_detection_confidence = minDetectionConfidence,
        min_tracking_confidence  = minTrackingConfidence,
        model_complexity         = modelComplexity
    )
    mpDrawing = mp.solutions.drawing_utils

    # Variables for FPS calculation
    prevTime = time.time()
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        success, frame = videoStream.read()
        if not success or frame is None: break
        
        # Preprocessing
        frame = cv2.flip(frame, 1)  # Flip horizontally
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and find hands using MediaPipe
        results = hands.process(rgbFrame)

        # Post-processing
        frame = cv2.cvtColor(rgbFrame, cv2.COLOR_RGB2BGR)

        # Calculate FPS
        currentTime = time.time()
        fps = 1 / (currentTime - prevTime) if (currentTime - prevTime) > 0 else 0
        prevTime = currentTime

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

                print(pointerData(handLandmarks))

        # Display the resulting frame
        cv2.imshow('Gesture Recognition', frame)

        # Exit Mechanism: Press any key to quit
        if cv2.waitKey(1) != -1: break

    # Cleanup
    videoStream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()