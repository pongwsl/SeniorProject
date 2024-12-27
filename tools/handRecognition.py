# # tools/handRecognition.py
# created by pongwsl on Dec 27, 2024
# latest edited on Dec 27, 2024
# to start camera and detect hand gesture/location

import cv2
import mediapipe as mp
import time
import threading
from typing import List, Tuple, Optional, Any

# Configuration Parameters
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_SKIP = 0  # Set to >0 to skip frames
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
MODEL_COMPLEXITY = 0  # 0 for lightweight, 1 for full

class VideoStream:
    """
    VideoStream class to capture video frames using a separate thread for optimized performance.
    """
    def __init__(self, src: int = 0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.ret, self.frame = self.capture.read()
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.capture.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self) -> Tuple[bool, Optional[Any]]:
        with self.lock:
            return self.ret, self.frame.copy() if self.ret else (False, None)

    def stop(self):
        self.running = False
        self.thread.join()
        self.capture.release()

class HandRecognition:
    """
    HandRecognition class using MediaPipe to detect hands and extract world landmarks.
    """
    def __init__(self,
                 max_num_hands: int = MAX_NUM_HANDS,
                 min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
                 min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
                 model_complexity: int = MODEL_COMPLEXITY):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity
        )
        self.mpDrawing = mp.solutions.drawing_utils

    def process_frame(self, frame: Any) -> Tuple[Any, List[Any]]:
        """
        Process a single frame to detect hands and return annotated frame and world landmarks.

        Args:
            frame: The BGR image frame to process.

        Returns:
            Annotated frame and list of world landmarks for each detected hand.
        """
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame and find hands
        results = self.hands.process(rgb_frame)
        # Convert back to BGR for OpenCV
        annotated_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        world_landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mpDrawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mpHands.HAND_CONNECTIONS,
                    self.mpDrawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mpDrawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                world_landmarks.append(hand_landmarks)
        
        return annotated_frame, world_landmarks

    def get_world_landmarks(self, frame: Any) -> List[Any]:
        """
        Extract world landmarks from a frame.

        Args:
            frame: The BGR image frame to process.

        Returns:
            List of world landmarks for each detected hand.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        world_landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                world_landmarks.append(hand_landmarks)
        return world_landmarks

    def close(self):
        """
        Release MediaPipe resources.
        """
        self.hands.close()

def main():
    """
    Main function for debugging HandRecognition.
    Captures video from webcam, processes each frame to detect hands, and displays the result with FPS.
    """
    # Initialize VideoStream
    video_stream = VideoStream(0)

    # Initialize HandRecognition
    hand_recognition = HandRecognition(
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        model_complexity=MODEL_COMPLEXITY
    )

    # Variables for FPS calculation
    prev_time = time.time()
    font = cv2.FONT_HERSHEY_SIMPLEX

    try:
        while True:
            success, frame = video_stream.read()
            if not success or frame is None:
                print("Failed to read frame from camera. Exiting...")
                break

            # Optional: Frame Skipping
            # Implement frame skipping logic here if FRAME_SKIP > 0
            # Example:
            # if FRAME_SKIP > 0:
            #     skip = FRAME_SKIP
            #     while skip > 0 and success:
            #         success, frame = video_stream.read()
            #         skip -= 1

            # Process the frame to detect hands and get annotated frame
            annotated_frame, world_landmarks = hand_recognition.process_frame(frame)

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time

            # Overlay FPS on the frame
            cv2.putText(
                annotated_frame, f'FPS: {fps:.2f}', (10, 30),
                font, 1, (0, 255, 0), 2, cv2.LINE_AA
            )

            # Optionally, display world landmarks coordinates
            if world_landmarks:
                for idx, hand in enumerate(world_landmarks):
                    for lm_id, landmark in enumerate(hand.landmark):
                        # Convert normalized coordinates to pixel values
                        h, w, _ = annotated_frame.shape
                        cx, cy, cz = int(landmark.x * w), int(landmark.y * h), landmark.z
                        cv2.putText(
                            annotated_frame, f'{lm_id}: ({cx}, {cy}, {cz:.2f})',
                            (10, 50 + idx * 20 + lm_id * 15),
                            font, 0.5, (255, 0, 0), 1, cv2.LINE_AA
                        )

            # Display the resulting frame
            cv2.imshow('Hand Recognition Debug', annotated_frame)

            # Exit Mechanism: Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exit key pressed. Exiting...")
                break

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")

    finally:
        # Cleanup
        hand_recognition.close()
        video_stream.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()