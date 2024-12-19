# test.py
# created by pongwsl on 18 dec 2024
# to test several libraries and to know which library should use on which python, venv.
# This code tested on python 3.9.6 / 3.12.8
# To select Python Interpreter in VSCode:
# press cmd+shift+P, type 'Python: Select Interpreter'
# use "/Library/Developer/CommandLineTools/usr/bin/python3"

import cv2
import mediapipe as mp
import time  # Import time module for FPS calculation

print('OpenCV version:', cv2.__version__)
print('Mediapipe version:', mp.__version__)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mpDrawing = mp.solutions.drawing_utils
camera = cv2.VideoCapture(0)

# Variables for FPS calculation
prev_time = 0
font = cv2.FONT_HERSHEY_SIMPLEX

while camera.isOpened():
    success, frame = camera.read()
    if not success:
        break

    # Flip the image horizontally for a later selfie-view display and also convert the color space from BGR to RGB
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

    # Process the frame and find hands using MediaPipe
    results = hands.process(frame)

    # Convert the color space back from RGB to BGR for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    prev_time = current_time

    # Overlay FPS on the frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            mpDrawing.draw_landmarks(frame, handLandmarks, mpHands.HAND_CONNECTIONS)

    cv2.imshow('Gesture Recognition', frame)  # Display the frame
    key = cv2.waitKey(1)
    if key != -1:
        break  # Press any key to quit

camera.release()
cv2.destroyAllWindows()