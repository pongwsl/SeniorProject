# tools/handControl.py
# created by pongwsl on Dec 27, 2024
# latest edited on Dec 27, 2024
# to start camera and detect hand gesture/location using MediaPipe

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

    prev_pos = None  # To store the previous position of the wrist

    try:
        while True:
            success, frame = videoStream.read()
            if not success or frame is None:
                print("Failed to read frame from camera. Exiting...")
                break

            # Process the frame to detect hands and get annotated frame and world landmarks
            annotatedFrame, worldLandmarks = handRecognition.processFrame(frame)

            if worldLandmarks:
                # Assuming we're tracking the first hand's wrist (landmark 0)
                wrist_landmark = worldLandmarks[0].landmark[0]
                current_pos = (wrist_landmark.x, wrist_landmark.y, wrist_landmark.z)

                if prev_pos is not None:
                    dx = current_pos[0] - prev_pos[0]
                    dy = current_pos[1] - prev_pos[1]
                    dz = current_pos[2] - prev_pos[2]
                else:
                    dx = dy = dz = 0.0  # No movement in the first frame

                prev_pos = current_pos

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