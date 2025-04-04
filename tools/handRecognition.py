# tools/handRecognition.py
# created by pongwsl on Dec 27, 2024
# latest edited on Dec 27, 2024
# to start camera and detect hand gesture/location using MediaPipe

import cv2
import mediapipe as mp
import time
import threading
from typing import List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats  # Import for mode calculation

# Configuration Parameters
frameWidth = 640
frameHeight = 480
frameSkip = 0  # Set to >0 to skip frames
maxNumHands = 1
minDetectionConfidence = 0.7
minTrackingConfidence = 0.7
modelComplexity = 1  # 0 for lightweight, 1 for full

class VideoStream:
    """
    VideoStream class to capture video frames using a separate thread for optimized performance.
    """
    def __init__(self, src: int = 0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
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
                 max_num_hands: int = maxNumHands,
                 min_detection_confidence: float = minDetectionConfidence,
                 min_tracking_confidence: float = minTrackingConfidence,
                 model_complexity: int = modelComplexity):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode = False,
            max_num_hands = max_num_hands,
            min_detection_confidence = min_detection_confidence,
            min_tracking_confidence = min_tracking_confidence,
            model_complexity = model_complexity
        )
        self.mpDrawing = mp.solutions.drawing_utils

    def processFrame(self, frame: Any) -> Tuple[Any, List[Any]]:
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
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame and find hands
        results = self.hands.process(rgbFrame)
        # Convert back to BGR for OpenCV
        annotatedFrame = cv2.cvtColor(rgbFrame, cv2.COLOR_RGB2BGR)

        # Draw normalized hand landmarks on the annotated frame
        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks:
                self.mpDrawing.draw_landmarks(
                    annotatedFrame,
                    handLandmarks,
                    self.mpHands.HAND_CONNECTIONS,
                    self.mpDrawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mpDrawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
        
        # Extract hand landmarks if available
        handLandmarks = []
        if results.multi_hand_landmarks:
            for handLandmark in results.multi_hand_landmarks:
                handLandmarks.append(handLandmark)

        return annotatedFrame, handLandmarks

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
    videoStream = VideoStream(0)

    # Initialize HandRecognition
    handRecognition = HandRecognition(
        max_num_hands = maxNumHands,
        min_detection_confidence = minDetectionConfidence,
        min_tracking_confidence = minTrackingConfidence,
        model_complexity = modelComplexity
    )

    # Variables for FPS calculation
    prevTime = time.time()
    font = cv2.FONT_HERSHEY_SIMPLEX
    latencies = []

    try:
        while True:
            success, frame = videoStream.read()
            if not success or frame is None:
                print("Failed to read frame from camera. Exiting...")
                break

            # Optional: Frame Skipping
            # Implement frame skipping logic here if frameSkip > 0
            # Example:
            # if frameSkip > 0:
            #     skip = frameSkip
            #     while skip > 0 and success:
            #         success, frame = videoStream.read()
            #         skip -= 1

            # Process the frame to detect hands and get annotated frame and world landmarks
            annotatedFrame, handLandmarks = handRecognition.processFrame(frame)

            # Calculate FPS and record latency
            currentTime = time.time()
            latency = (currentTime - prevTime) * 1000  # Convert to milliseconds
            latencies.append(latency)
            fps = 1 / (currentTime - prevTime) if (currentTime - prevTime) > 0 else 0
            prevTime = currentTime

            # Overlay FPS on the frame
            cv2.putText(
                annotatedFrame, f'FPS: {fps:.2f}', (10, 30),
                font, 1, (0, 255, 0), 2, cv2.LINE_AA
            )
            cv2.putText(
                annotatedFrame, f'Frames: {len(latencies)}', (10, 60),
                font, 1, (0, 255, 255), 2, cv2.LINE_AA
            )

            # Optionally, display world landmarks coordinates
            if handLandmarks:
                # each hand (which we should have only one hand in the frame for this project)
                for idx, hand in enumerate(handLandmarks):
                    # each landmark
                    for lm_id, landmark in enumerate(hand.landmark):
                        x, y, z = landmark.x, landmark.y, landmark.z
                        cv2.putText(
                            annotatedFrame, f'{lm_id}: ({x:.2f}, {y:.2f}, {z:.2f})',
                            (10, 50 + idx * 20 + lm_id * 15),
                            font, 0.5, (255, 0, 0), 1, cv2.LINE_AA
                        )

            # Display the resulting frame
            cv2.imshow('Hand Recognition Debug', annotatedFrame)

            # Exit Mechanism: Press any key to quit
            if cv2.waitKey(1) != -1:
                print("Exit key pressed. Exiting...")
                break
        
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")

    finally:
        # Cleanup
        handRecognition.close()
        videoStream.stop()
        cv2.destroyAllWindows()
        total_frames = len(latencies)
        avg_latency = np.mean(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        range_latency = max_latency - min_latency
        median_latency = np.median(latencies)
        mode_latency = stats.mode(latencies, keepdims=True)[0][0]
        std_dev_latency = np.std(latencies)

        print(f"Total Frames Processed: {total_frames}")
        print(f"Average Latency: {avg_latency:.2f} ms")
        print(f"Min Latency: {min_latency:.2f} ms")
        print(f"Max Latency: {max_latency:.2f} ms")
        print(f"Latency Range: {range_latency:.2f} ms")
        print(f"Median Latency: {median_latency:.2f} ms")
        print(f"Mode Latency: {mode_latency:.2f} ms")
        print(f"Standard Deviation: {std_dev_latency:.2f} ms")

        barWidth = 0.2
        # Plot Latency Histogram
        latency_bins = np.arange(int(min(latencies)), int(max(latencies)) + barWidth, barWidth)  # +2 to include the last bin
        latency_counts, _ = np.histogram(latencies, bins=latency_bins)

        plt.figure(figsize=(10, 5))
        plt.bar(latency_bins[:-1], latency_counts, width=barWidth, align='edge', color='blue', edgecolor='black')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Number of Frames')
        plt.title('handRecognition.py Frame Processing Latency Distribution')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

if __name__ == "__main__":
    main()