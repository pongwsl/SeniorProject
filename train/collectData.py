# collectData.py
# Created by pongwsl on 23 Dec 2024
# Last updated 24 Dec 2024
# Script to collect hand landmarks and label pinching actions

import cv2
import mediapipe as mp
import csv
import os

def initializeCSV(filePath):
    """Initialize the CSV file with headers if it doesn't exist."""
    if not os.path.exists(filePath):
        with open(filePath, mode='w', newline='') as file:
            writer = csv.writer(file)
            # 21 landmarks each with x, y, z
            header = []
            for i in range(21):
                header.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
            header.append('label')
            writer.writerow(header)

def writeLandmarks(filePath, landmarks, label):
    """Write the landmarks and label to the CSV file."""
    with open(filePath, mode='a', newline='') as file:
        writer = csv.writer(file)
        row = []
        for lm in landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z])
        row.append(label)
        writer.writerow(row)

def collect_data(label, hands, mpHands, mpDrawing, cap, filePath, totalFrames=1000):
    """Collect a specified number of frames with the given label."""
    collectedFrames = 0
    print(f"=== Collecting '{label}' frames ===")
    print("Press 'q' to quit data collection early.")

    while collectedFrames < totalFrames:
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame. Exiting data collection.")
            break

        # Flip the frame horizontally for a mirror view
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        results = hands.process(rgbFrame)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks:
                mpDrawing.draw_landmarks(
                    frame, handLandmarks, mpHands.HAND_CONNECTIONS,
                    mpDrawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mpDrawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
        else:
            # Optional: Indicate no hands detected
            cv2.putText(
                frame, "No Hands Detected", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA
            )

        # Display the frame
        cv2.imshow('Data Collection', frame)

        # Write landmarks if hands are detected
        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks:
                writeLandmarks(filePath, handLandmarks, label)
                collectedFrames += 1
                print(f"Collected Frames: {collectedFrames}/{totalFrames}")
                break  # Only one hand per frame as per max_num_hands=1
        else:
            print("No hands detected. Skipping frame.")

        # Check for 'q' key press to quit data collection
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting data collection early.")
            break

    print(f"=== '{label}' Data Collection Completed ===")

def main():
    # Configuration Parameters
    filePath = 'data.csv'
    initializeCSV(filePath)

    # Initialize MediaPipe Hands
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
        model_complexity=0
    )
    mpDrawing = mp.solutions.drawing_utils

    # Initialize Video Capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("=== Data Collection Started ===")
    print("Press 'o' for 'on' (Pinching), 'f' for 'off' (Not Pinching), or 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame. Exiting...")
            break

        # Flip the frame horizontally for a mirror view
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        results = hands.process(rgbFrame)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks:
                mpDrawing.draw_landmarks(
                    frame, handLandmarks, mpHands.HAND_CONNECTIONS,
                    mpDrawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mpDrawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
        else:
            # Optional: Indicate no hands detected
            cv2.putText(
                frame, "No Hands Detected", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA
            )

        # Display the frame
        cv2.imshow('Data Collection', frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('o'):
            # Start collecting 'on' labeled data
            collect_data('on', hands, mpHands, mpDrawing, cap, filePath, totalFrames=1000)
        elif key == ord('f'):
            # Start collecting 'off' labeled data
            collect_data('off', hands, mpHands, mpDrawing, cap, filePath, totalFrames=1000)
        elif key == ord('q'):
            print("Quitting data collection.")
            break
        else:
            # Continue displaying the feed without collecting data
            continue

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("=== Data Collection Terminated ===")

if __name__ == "__main__":
    main()