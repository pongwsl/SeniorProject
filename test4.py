# test4.py
# created by pongwsl on 23 Dec 2024
# last updated 23 Dec 2024
# To test MediaPipe Hand detection and its accuracy
# and also to test tools/pointerData.py
# Use picture from data/ folder

from tools.pointerData import pointerData, pointerWorldData
import cv2
import mediapipe as mp
import os
import glob
import math
from tabulate import tabulate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def print_normalized_landmarks(handLandmarks):
    headers = ["Landmark", "x ()", "y ()", "z ()"]
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

def plot_world_landmarks(worldLandmarks, hand_idx, img_name):
    """
    Plots the world landmarks in a 3D rotatable space.
    
    Args:
        worldLandmarks: The world landmarks from MediaPipe.
        hand_idx: Index of the hand being plotted.
        img_name: Name of the image being processed.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract x, y, z coordinates and convert them to millimeters for better visualization
    x = [lm.x * 1000 for lm in worldLandmarks.landmark]
    y = [lm.y * 1000 for lm in worldLandmarks.landmark]
    z = [lm.z * 1000 for lm in worldLandmarks.landmark]
    
    # Plot the landmarks
    ax.scatter(x, y, z, c='b', marker='o')
    
    # Optionally, connect the landmarks with lines to represent the hand skeleton
    connections = mp.solutions.hands.HAND_CONNECTIONS
    for connection in connections:
        start_idx, end_idx = connection
        ax.plot([x[start_idx], x[end_idx]], [y[start_idx], y[end_idx]], [z[start_idx], z[end_idx]], c='r')
    
    # Set labels
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Hand {hand_idx} - {img_name}')
    
    # Set equal aspect ratio for all axes
    max_range = max(max(x) - min(x), max(y) - min(y), max(z) - min(z)) / 2.0
    mid_x = (max(x) + min(x)) * 0.5
    mid_y = (max(y) + min(y)) * 0.5
    mid_z = (max(z) + min(z)) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # plt.show()

    # Rotate the axes and update
    for angle in range(0, 360*4 + 1):
        # Normalize the angle to the range [-180, 180] for display
        angle_norm = (angle + 180) % 360 - 180

        # Cycle through a full rotation of elevation, then azimuth, roll, and all
        elev = azim = roll = 0
        if angle <= 360:
            elev = angle_norm
        elif angle <= 360*2:
            azim = angle_norm
        elif angle <= 360*3:
            roll = angle_norm
        else:
            elev = azim = roll = angle_norm

        # Update the axis view and title
        ax.view_init(elev, azim, roll)
        plt.title('Elevation: %d°, Azimuth: %d°, Roll: %d°' % (elev, azim, roll))

        plt.draw()
        plt.pause(.001)

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
        max_num_hands = 1,               # Adjust as needed
        min_detection_confidence = 0.5,
        model_complexity = 1             # 0 for lightweight, 1 for full
    )
    mpDrawing = mp.solutions.drawing_utils


# Get list of all PNG and JPG/JPEG files in the data folder
    imageExtensions = ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']
    imagePaths = [img for ext in imageExtensions for img in glob.glob(os.path.join(dataFolder, ext))]

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
                mpDrawing.draw_landmarks(
                    image, handLandmarks, mpHands.HAND_CONNECTIONS,
                    mpDrawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mpDrawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

                # Process and print normalized landmarks
                print(f"\nHand {idx} - Pointer Data (Normalized):")
                handData = pointerData(handLandmarks)
                print(handData)

                # Print normalized landmarks with units
                print_normalized_landmarks(handLandmarks)

                # Process and print world landmarks
                print(f"\nHand {idx} - Pointer Data (World):")
                worldData = pointerWorldData(worldLandmarks)
                print(worldData)

                # Print world landmarks with units
                print_world_landmarks(worldLandmarks)

                # Plot world landmarks in 3D
                img_name = os.path.basename(imgPath)
                plot_world_landmarks(worldLandmarks, idx, img_name)

                # Debug print before showing the image
                print(f"Displaying image: {os.path.basename(imgPath)}")
                cv2.imshow('Hand Landmarks', image)
                cv2.waitKey(0)  # Wait indefinitely until a key is pressed
                print(f"Closed image: {os.path.basename(imgPath)}")

        else:
            print(f"\nNo hands detected in image: {os.path.basename(imgPath)}")

    # Cleanup
    hands.close()
    # cv2.destroyAllWindows()  # Uncomment if you display images

if __name__ == "__main__":
    main()