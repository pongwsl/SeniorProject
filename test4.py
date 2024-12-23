# test4.py
# created by pongwsl on 23 Dec 2024
# last updated 23 Dec 2024
# To test tools/pointerData.py

from tools.pointerData import pointerData, pointerWorldData
import cv2
import mediapipe as mp
import os
import glob

def main():
    # Path to the folder containing PNG images
    dataFolder = os.path.join(os.path.dirname(__file__), 'data')

    # Check if data folder exists
    if not os.path.exists(dataFolder):
        print(f"Data folder not found at: {dataFolder}")
        return

    # Initialize MediaPipe Hands with world landmarks
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode = True,        # Since we're processing static images
        max_num_hands = 2,               # Adjust as needed
        min_detection_confidence = 0.8,
        min_tracking_confidence = 0.8,
        model_complexity = 0             # 0 for lightweight, 1 for full
    )
    mpDrawing = mp.solutions.drawing_utils

    # Get list of all PNG files in the data folder
    imagePaths = glob.glob(os.path.join(dataFolder, '*.png'))

    if not imagePaths:
        print(f"No PNG images found in the data folder: {dataFolder}")
        return

    # Iterate through each image
    for imgPath in imagePaths:
        # Read the image
        image = cv2.imread(imgPath)

        if image is None:
            print(f"Failed to read image: {imgPath}")
            continue

        # Convert the BGR image to RGB as MediaPipe uses RGB
        rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and detect hands
        results = hands.process(rgbImage)

        # Check if any hands are detected
        if results.multi_hand_landmarks and results.multi_hand_world_landmarks:
            print(f"\nProcessing Image: {os.path.basename(imgPath)}")

            # Iterate through each detected hand
            for idx, (handLandmarks, worldLandmarks) in enumerate(zip(results.multi_hand_landmarks, results.multi_hand_world_landmarks), start=1):
                # Optionally, draw landmarks on the image (uncomment if needed)
                # mpDrawing.draw_landmarks(
                #     image, handLandmarks, mpHands.HAND_CONNECTIONS,
                #     mpDrawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                #     mpDrawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                # )
                print(idx, worldLandmarks)
                

            # Optionally, display the image with landmarks (uncomment if needed)
            # cv2.imshow('Hand Landmarks', image)
            # cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        else:
            print(f"\nNo hands detected in image: {os.path.basename(imgPath)}")

    # Cleanup
    hands.close()
    # cv2.destroyAllWindows()  # Uncomment if you display images

if __name__ == "__main__":
    main()