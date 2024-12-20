# test2b.py
# created by pongwsl on 20 dec 2024
# lasted updated 20 dec 2024
# update from test1.py, to optimize the MediaPipe.
# This code tested on python 3.9.6 / 3.12.8
# To select Python Interpreter in VSCode:
# press cmd+shift+P, type 'Python: Select Interpreter'
# use "/Library/Developer/CommandLineTools/usr/bin/python3"

import cv2
import mediapipe as mp
import time
import multiprocessing as mp_process
from multiprocessing import Queue, Event

# Configuration Parameters
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MAX_QUEUE_SIZE = 10
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
MODEL_COMPLEXITY = 0  # 0 for lightweight, 1 for full

def frame_capture(queue, stop_event):
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    while not stop_event.is_set():
        ret, frame = camera.read()
        if not ret:
            continue
        if not queue.full():
            queue.put(frame)
    camera.release()

def frame_processor(queue, stop_event):
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        model_complexity=MODEL_COMPLEXITY
    )
    mpDrawing = mp.solutions.drawing_utils
    
    prev_time = time.time()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    while not stop_event.is_set():
        if not queue.empty():
            frame = queue.get()
            # Preprocessing
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe
            results = hands.process(rgb_frame)
            
            # Post-processing
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            
            # Overlay FPS
            cv2.putText(
                frame, f'FPS: {fps:.2f}', (10, 30),
                font, 1, (0, 255, 0), 2, cv2.LINE_AA
            )
            
            # Draw Hand Landmarks
            if results.multi_hand_landmarks:
                for handLandmarks in results.multi_hand_landmarks:
                    mpDrawing.draw_landmarks(
                        frame, handLandmarks, mpHands.HAND_CONNECTIONS,
                        mpDrawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mpDrawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
            
            # Display Frame
            cv2.imshow('Gesture Recognition', frame)
            
            # Exit Mechanism: Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    frame_queue = Queue(maxsize=MAX_QUEUE_SIZE)
    stop_event = Event()
    
    # Initialize Processes
    capture_process = mp_process.Process(target=frame_capture, args=(frame_queue, stop_event))
    processor_process = mp_process.Process(target=frame_processor, args=(frame_queue, stop_event))
    
    # Start Processes
    capture_process.start()
    processor_process.start()
    
    try:
        while processor_process.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
    
    # Ensure Processes Terminate
    capture_process.join()
    processor_process.join()