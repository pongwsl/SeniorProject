# # tools/handControl.py
# created by pongwsl on Dec 27, 2024
# latest edited on Dec 27, 2024
# to control object in CoppeliaSim using MediaPipe

import cv2
from tools.handRecognition import HandRecognition, VideoStream

def main():
    # Initialize VideoStream
    video_stream = VideoStream(0)

    # Initialize HandRecognition
    hand_recognition = HandRecognition()

    while True:
        success, frame = video_stream.read()
        if not success:
            break

        annotated_frame, landmarks = hand_recognition.process_frame(frame)

        # Use the landmarks as needed
        # For example, print the coordinates of the first landmark of the first hand
        if landmarks:
            first_landmark = landmarks[0].landmark[0]
            print(f'First Landmark: x={first_landmark.x}, y={first_landmark.y}, z={first_landmark.z}')

        cv2.imshow('Main Hand Recognition', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    hand_recognition.close()
    video_stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()