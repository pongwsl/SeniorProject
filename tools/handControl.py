# # tools/handControl.py
# created by pongwsl on Dec 27, 2024
# latest edited on Dec 27, 2024
# to control object in CoppeliaSim using MediaPipe

import cv2
from handRecognition import HandRecognition, VideoStream

def main():
    # Initialize VideoStream
    videoStream = VideoStream(0)

    # Initialize HandRecognition
    handRecognition = HandRecognition()

    while True:
        success, frame = videoStream.read()
        if not success:
            break

        annotatedFrame, worldLandmarks = handRecognition.processFrame(frame)

        # Use the worldLandmarks as needed
        # Example: Print the coordinates of the first landmark of the first hand
        if worldLandmarks:
            firstLandmark = worldLandmarks[0].landmark[0]
            print(f'First Landmark: x={firstLandmark.x:.2f}, y={firstLandmark.y:.2f}, z={firstLandmark.z:.2f}')

        cv2.imshow('Main Hand Recognition', annotatedFrame)

        if cv2.waitKey(1) != -1:
            break

    # Cleanup
    handRecognition.close()
    videoStream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()