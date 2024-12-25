# test5.py
# created by pongwsl on 24 Dec 2024
# last updated 24 Dec 2024
# To compare CPU and GPU MediaPipe running using pictures from data folder

import cv2
import mediapipe as mp
import os
import glob
import time
from tabulate import tabulate

def print_normalized_landmarks(handLandmarks):
    headers = ["Landmark", "x (normalized)", "y (normalized)", "z (normalized)"]
    table = []
    for idx, lm in enumerate(handLandmarks.landmark):
        table.append([idx, f"{lm.x:.3f}", f"{lm.y:.3f}", f"{lm.z:.3f}"])
    print(tabulate(table, headers=headers, tablefmt="grid"))

def print_world_landmarks(worldLandmarks):
    headers = ["Landmark", "x (mm)", "y (mm)", "z (mm)"]
    table = []
    for idx, lm in enumerate(worldLandmarks.landmark):
        table.append([idx, f"{lm.x * 1000:.3f}", f"{lm.y * 1000:.3f}", f"{lm.z * 1000:.3f}"])
    print(tabulate(table, headers=headers, tablefmt="grid"))

def main():
    # Path to the folder containing PNG and JPG/JPEG images
    dataFolder = os.path.join(os.path.dirname(__file__), 'data')

    # Check if data folder exists
    if not os.path.exists(dataFolder):
        print(f"Data folder not found at: {dataFolder}")
        return

    # Initialize MediaPipe Hands with world landmarks (CPU)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=True,        # Since we're processing static images
        max_num_hands=2,               # Adjust as needed
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
        model_complexity=0             # 0 for lightweight, 1 for full
    )
    mpDrawing = mp.solutions.drawing_utils

    # Get list of all PNG and JPG/JPEG files in the data folder
    imageExtensions = ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']
    imagePaths = [img for ext in imageExtensions for img in glob.glob(os.path.join(dataFolder, ext))]

    if not imagePaths:
        print(f"No PNG or JPG/JPEG images found in the data folder: {dataFolder}")
        return

    # Initialize lists to store processing times
    processing_times_cpu = []

    # Iterate through each image
    for imgPath in imagePaths:
        # Read the image
        image = cv2.imread(imgPath)

        if image is None:
            print(f"Failed to read image: {imgPath}")
            continue

        # Convert the BGR image to RGB as MediaPipe uses RGB
        rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Start time measurement for CPU processing
        start_time_cpu = time.time()

        # Process the image and detect hands
        results = hands.process(rgbImage)

        # End time measurement for CPU processing
        end_time_cpu = time.time()

        # Calculate processing time
        processing_time_cpu = end_time_cpu - start_time_cpu
        processing_times_cpu.append(processing_time_cpu)

        # Check if any hands are detected
        if results.multi_hand_landmarks and results.multi_hand_world_landmarks:
            print(f"\nProcessing Image: {os.path.basename(imgPath)}")
            print(f"CPU Processing Time: {processing_time_cpu:.4f} seconds")

            # Iterate through each detected hand
            for idx, (handLandmarks, worldLandmarks) in enumerate(zip(results.multi_hand_landmarks, results.multi_hand_world_landmarks), start=1):
                # Draw landmarks on the image (optional, commenting out since no imshow)
                # mpDrawing.draw_landmarks(
                #     image, handLandmarks, mpHands.HAND_CONNECTIONS,
                #     mpDrawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                #     mpDrawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                # )

                # Process and print normalized landmarks
                print(f"\nHand {idx} - Pointer Data (Normalized):")
                # Assuming pointerData is a function that processes handLandmarks
                # Replace the following line with actual processing if needed
                handData = "Placeholder for pointerData output"
                print(handData)

                # Print normalized landmarks with units
                print_normalized_landmarks(handLandmarks)

                # Process and print world landmarks
                print(f"\nHand {idx} - Pointer Data (World):")
                # Assuming pointerWorldData is a function that processes worldLandmarks
                # Replace the following line with actual processing if needed
                worldData = "Placeholder for pointerWorldData output"
                print(worldData)

                # Print world landmarks with units
                print_world_landmarks(worldLandmarks)

        else:
            print(f"\nNo hands detected in image: {os.path.basename(imgPath)}")
            print(f"CPU Processing Time: {processing_time_cpu:.4f} seconds")

    # Calculate and display summary statistics
    if processing_times_cpu:
        avg_time_cpu = sum(processing_times_cpu) / len(processing_times_cpu)
        max_time_cpu = max(processing_times_cpu)
        min_time_cpu = min(processing_times_cpu)

        print("\n--- CPU Processing Time Summary ---")
        print(f"Total Images Processed: {len(processing_times_cpu)}")
        print(f"Average Processing Time: {avg_time_cpu:.4f} seconds")
        print(f"Maximum Processing Time: {max_time_cpu:.4f} seconds")
        print(f"Minimum Processing Time: {min_time_cpu:.4f} seconds")
    else:
        print("\nNo images were processed.")

    # Cleanup
    hands.close()

if __name__ == "__main__":
    main()