# tools/handControl.py
# created by pongwsl on Dec 27, 2024
# latest edited on Feb 21, 2025
#
# Uses handRecognition.py to get dx, dy from index finger movement,
# and calculates dz from the change in distance between landmarks 0 and 9
# to indicate how far the hand is from the camera.

import cv2
import time
import math
from typing import Tuple, Optional

# When running this module directly, adjust sys.path to ensure the parent folder is in the path.
if __name__ == "__main__" and __package__ is None:
    import sys
    import os
    # Add the parent directory to sys.path so that 'tools' is recognized.
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "tools"
from .handRecognition import HandRecognition, VideoStream

def handControl():
    """
    Detects hand movement and yields the delta (dx, dy, dz).

    - dx, dy: derived from the difference in the index fingertip (landmark 8)
              between consecutive frames.
    - dz:     derived from how the distance between landmarks 0 and 9 changes,
              indicating hand moving closer or farther from the camera.

    Yields:
        Tuple[float, float, float]: (dx, dy, dz)
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

    # To store the previous frame's index-finger position and hand size
    prevFingerPos = None
    prevHandSize = None

    # A scale factor to amplify/reduce dz changes based on hand size
    dzScaleFactor = 1.0

    try:
        while True:
            success, frame = videoStream.read()
            if not success or frame is None:
                print("Failed to read frame from camera. Exiting...")
                break

            # Process the frame to detect hands and get annotated frame and world landmarks
            annotatedFrame, handLandmarks = handRecognition.processFrame(frame)

            if handLandmarks:
                # Extract the index-finger tip (landmark 8)
                indexTip = handLandmarks[0].landmark[8]
                currentFingerPos = (indexTip.x, indexTip.y, indexTip.z)

                # Compute dx, dy from the index-finger tip movement
                if prevFingerPos is not None:
                    dx = currentFingerPos[0] - prevFingerPos[0]
                    dy = currentFingerPos[1] - prevFingerPos[1]
                else:
                    dx = 0.0
                    dy = 0.0

                # Update for next iteration
                prevFingerPos = currentFingerPos

                # -- Compute hand size (distance between landmarks 0 and 9) --
                wrist = handLandmarks[0].landmark[0]
                palm  = handLandmarks[0].landmark[9]
                # 3D distance between wrist (0) and palm (9):
                currentHandSize = math.sqrt(
                    (wrist.x - palm.x)**2 +
                    (wrist.y - palm.y)**2 +
                    (wrist.z - palm.z)**2
                )

                # If we have a previous hand size, compute dz from the size change
                if prevHandSize is not None:
                    # If the hand becomes smaller, we interpret that as moving away => +dz
                    sizeDiff = (prevHandSize - currentHandSize)
                    dz = dzScaleFactor * sizeDiff
                else:
                    dz = 0.0

                prevHandSize = currentHandSize
            else:
                dx, dy, dz = 0.0, 0.0, 0.0

            # Overlay the movement values on the annotated frame.
            text = f"dx: {dx:.4f}, dy: {dy:.4f}, dz: {dz:.4f}"
            cv2.putText(annotatedFrame, text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Display the annotated frame for debugging purposes.
            cv2.imshow('Hand Control', annotatedFrame)

            # Exit mechanism: Press any key to exit.
            if cv2.waitKey(1) != -1:
                print("Key pressed. Exiting...")
                break

            yield (dx, dy, dz)

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")

    finally:
        # Cleanup resources
        handRecognition.close()
        videoStream.stop()
        cv2.destroyAllWindows()


def main():
    """
    Main function for debugging handControl().
    The (dx, dy, dz) values are now displayed on the imshow window instead of printing to the console.
    """
    for _ in handControl():
        # The loop is solely to run handControl and display the video feed.
        pass


if __name__ == "__main__":
    main()