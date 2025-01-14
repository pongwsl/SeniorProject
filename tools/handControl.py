# tools/handControl.py
# created by pongwsl on Dec 27, 2024
# latest edited on Dec 27, 2024
# to use handRecognition.py to get dx, dy, dz

import cv2
import time
from typing import Tuple, Optional

from handRecognition import HandRecognition, VideoStream

def handControl():
    """
    Detects hand movement and yields the delta (dx, dy, dz) of the wrist for each frame.

    Yields:
        Tuple[float, float, float]: The change in x, y, z coordinates since the last frame.
    """
    # Initialize VideoStream
    videoStream = VideoStream(0)

    # Initialize HandRecognition
    handRecognition = HandRecognition(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        model_complexity=0
    )

    prevPos = None  # To store the previous position of the wrist

    try:
        while True:
            success, frame = videoStream.read()
            if not success or frame is None:
                print("Failed to read frame from camera. Exiting...")
                break

            # Process the frame to detect hands and get annotated frame and world landmarks
            annotatedFrame, handLandmarks = handRecognition.processFrame(frame)

            if handLandmarks:
                pointLandmark = handLandmarks[0].landmark[8] # first hand, landmark no. 8
                currentPos = (pointLandmark.x, pointLandmark.y, pointLandmark.z)

                if prevPos is not None:
                    dx = currentPos[0] - prevPos[0]
                    dy = currentPos[1] - prevPos[1]
                    dz = currentPos[2] - prevPos[2]
                else:
                    dx = dy = dz = 0.0  # No movement in the first frame

                prevPos = currentPos

                yield (dx, dy, dz)
            else:
                # No hand detected; yield zero movement
                yield (0.0, 0.0, 0.0)

            # Display the annotated frame for debugging purposes
            cv2.imshow('Hand Control', annotatedFrame)

            # Exit Mechanism: Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exit key pressed. Exiting...")
                break

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")

    finally:
        # Cleanup resources
        handRecognition.close()
        videoStream.stop()
        cv2.destroyAllWindows()

def main():
    """
    Main function for debugging handControl.
    Continuously prints the (dx, dy, dz) output from handControl().
    """

    for dx, dy, dz in handControl():
        print(f"dx: {dx:.2f}, dy: {dy:.2f}, dz: {dz:.2f}")

if __name__ == "__main__":
    main()