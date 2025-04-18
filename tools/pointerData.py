# tools/pointerData.py
# created by pongwsl on 23 dec 2024
# last updated 18 apr 2025
# use getPosition.py

# When running this module directly, adjust sys.path to ensure the parent folder is in the path.
if __name__ == "__main__" and __package__ is None:
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "tools"
from .handRecognition import HandRecognition, VideoStream

import cv2
import mediapipe as mp
import time
import math

# Configuration Parameters
frameWidth = 640
frameHeight = 480
frameSkip = 0  # Set to >0 to skip frames
maxNumHands = 1
minDetectionConfidence = 0.8  # higher is faster (GPT's opinion)
minTrackingConfidence = 0.8
modelComplexity = 0  # 0 for lightweight, 1 for full

def link(lm, n, m, a):
    return getattr(lm.landmark[n], a) - getattr(lm.landmark[m], a)

def length(lm, n, m):
    return math.sqrt(
        link(lm, n, m, 'x') ** 2 +
        link(lm, n, m, 'y') ** 2 +
        link(lm, n, m, 'z') ** 2
    )

def pointerData(lm):
    try:
        rollAngle = math.degrees(math.atan(link(lm, 5, 17, 'y') / link(lm, 5, 17, 'x')))
        pitchAngle = math.degrees(math.asin(link(lm, 8, 5, 'y') / length(lm, 8, 5)))
        yawAngle = math.degrees(math.asin(link(lm, 8, 5, 'x') / length(lm, 8, 5)))

        # Pinching Action
        pinchingAction = 'Nozzle off'
        calibator = 1
        if (length(lm, 8, 4) * calibator) < length(lm, 4, 3): 
            pinchingAction = 'NOZZLE ON '
        elif (length(lm, 8, 3) * calibator) < length(lm, 4, 3): 
            pinchingAction = 'NOZZLE ON '
        elif (length(lm, 8, 2) * calibator) < length(lm, 4, 3): 
            pinchingAction = 'NOZZLE ON '
        elif (length(lm, 8, 1) * calibator) < length(lm, 4, 3): 
            pinchingAction = 'NOZZLE ON '

        # Return
        return (
            pinchingAction, 
            round(rollAngle, 2), 
            round(pitchAngle, 2), 
            round(yawAngle, 2), 
            round(length(lm, 4, 3), 3), 
            round(length(lm, 8, 4), 3)
        )
    except ZeroDivisionError:
        return ("Undefined angles due to division by zero", 0, 0, 0, 0, 0)
    
def pointerWorldData(lm):
    try:
        rollAngle = math.degrees(math.atan(link(lm, 5, 17, 'y') / link(lm, 5, 17, 'x')))
        pitchAngle = math.degrees(math.asin(link(lm, 8, 5, 'y') / length(lm, 8, 5)))
        yawAngle = math.degrees(math.asin(link(lm, 8, 5, 'x') / length(lm, 8, 5)))

        # Pinching Action
        calibator = 1.5
        if (length(lm, 8, 4) * calibator) < length(lm, 4, 3): 
            pinchingAction = 'NOZZLE ON '
        else:
            pinchingAction = 'Nozzle off'
        
        return (
            pinchingAction, 
            round(rollAngle, 2), 
            round(pitchAngle, 2), 
            round(yawAngle, 2), 
            round(length(lm, 4, 3), 3), 
            round(length(lm, 8, 4), 3)
        )
    except ZeroDivisionError:
        return ("Undefined angles due to division by zero", 0, 0, 0, 0, 0)

def main():
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

    n = 0
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
        if results.multi_hand_landmarks and results.multi_hand_world_landmarks:
            for handLandmarks, worldLandmarks in zip(results.multi_hand_landmarks, results.multi_hand_world_landmarks):
                # Draw normalized hand landmarks
                mpDrawing.draw_landmarks(
                    frame, handLandmarks, mpHands.HAND_CONNECTIONS,
                    mpDrawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mpDrawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

                # Process and print normalized landmarks
                print("Pointer Data (Normalized):", pointerData(handLandmarks))

                # Process and print world landmarks
                # worldData = pointerData(worldLandmarks)
                # print(f"Pointer Data (World) xxx : {worldData}")

        # Display the resulting frame
        cv2.imshow('Gesture Recognition', frame)

        # Exit Mechanism: Press any key to quit
        if cv2.waitKey(1) != -1: break

    # Cleanup
    videoStream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()