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

        annotatedFrame, landmarks = handRecognition.processFrame(frame)

        # Use the landmarks as needed
        # For example, print the coordinates of the first landmark of the first hand
        if landmarks:
            first_landmark = landmarks[0].landmark[0]
            print(f'First Landmark: x={first_landmark.x}, y={first_landmark.y}, z={first_landmark.z}')

        cv2.imshow('Main Hand Recognition', annotatedFrame)

        if cv2.waitKey(1) != -1:
            break

    # Cleanup
    handRecognition.close()
    videoStream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()