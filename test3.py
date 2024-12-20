# test3.py
# created by pongwsl on 20 dec 2024
# lasted updated 20 dec 2024
# update from test2a.py, to add def tipCoord(lm, i).
# This code tested on python 3.9.13 on venv-metal.
# To select Python Interpreter in VSCode:
# press cmd+shift+P, type 'Python: Select Interpreter'
# use "/Library/Developer/CommandLineTools/usr/bin/python3"

import cv2
import mediapipe as mp
import time
import threading
import math

# Configuration Parameters
frameWidth = 640
frameHeight = 480
frameSkip = 0  # Set to >0 to skip frames
maxNumHands = 1
minDetectionConfidence = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
MODEL_COMPLEXITY = 0  # 0 for lightweight, 1 for full

class VideoStream:
    def __init__(self, src=0):
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

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.ret else (False, None)

    def stop(self):
        self.running = False
        self.thread.join()
        self.capture.release()

def tipCoord(lm, i): #getTipCoordinates

    fingerList = ["Thumb","Index","Middle","Ring","Pinky"]
    fingerName = fingerList[i]

    tipNo = i*4+4
    tipCoordinates = (lm.landmark[tipNo].x,lm.landmark[tipNo].y)
    roundedTipCoord = (round(lm.landmark[tipNo].x, 4), round(lm.landmark[tipNo].y, 4))

    return (fingerName, roundedTipCoord)

def pointerData(lm): # Nozzle on/off

    x0Wrist = lm.landmark[0].x
    y0Wrist = lm.landmark[0].y
    z0Wrist = lm.landmark[0].z

    # Thumb coordinates
    x4ThumbTip = lm.landmark[4].x
    y4ThumbTip = lm.landmark[4].y
    z4ThumbTip = lm.landmark[4].z

    x3ThumbTip = lm.landmark[3].x
    y3ThumbTip = lm.landmark[3].y
    z3ThumbTip = lm.landmark[3].z

    x2ThumbMcp = lm.landmark[2].x
    y2ThumbMcp = lm.landmark[2].y
    z2ThumbMcp = lm.landmark[2].z

    x1ThumbCmc = lm.landmark[1].x
    y1ThumbCmc = lm.landmark[1].y
    z1ThumbCmc = lm.landmark[1].z

    # Index coordinates
    x8IndexTip = lm.landmark[8].x
    y8IndexTip = lm.landmark[8].y
    z8IndexTip = lm.landmark[8].z

    x7IndexDip = lm.landmark[7].x
    y7IndexDip = lm.landmark[7].y
    z7IndexDip = lm.landmark[7].z

    x6IndexPip = lm.landmark[6].x
    y6IndexPip = lm.landmark[6].y
    z6IndexPip = lm.landmark[6].z

    x5IndexMcp = lm.landmark[5].x
    y5IndexMcp = lm.landmark[5].y
    z5IndexMcp = lm.landmark[5].z

    #MISC
    x17PinkyMcp = lm.landmark[17].x
    y17PinkyMcp = lm.landmark[17].y
    z17PinkyMcp = lm.landmark[17].z

    #Knuckle Lengths
        # Thumb
    Link43 = ((x4ThumbTip-x3ThumbTip)**2 + (y4ThumbTip-y3ThumbTip)**2 + (z4ThumbTip-z3ThumbTip)**2)**0.5 # D Knuckle
    Link32 = ((x3ThumbTip-x2ThumbMcp)**2 + (y3ThumbTip-y2ThumbMcp)**2 + (z3ThumbTip-z2ThumbMcp)**2)**0.5 # C Knuckle

        # Index
    Link87 = ((x8IndexTip-x7IndexDip)**2 + (y8IndexTip-y7IndexDip)**2 + (z8IndexTip-z7IndexDip)**2)**0.5 # H Knuckle
    Link76 = ((x7IndexDip-x6IndexPip)**2 + (y7IndexDip-y6IndexPip)**2 + (z7IndexDip-z6IndexPip)**2)**0.5 # G Knuckle
    Link65 = ((x6IndexPip-x5IndexMcp)**2 + (y6IndexPip-y5IndexMcp)**2 + (z6IndexPip-z5IndexMcp)**2)**0.5 # F Knuckle
        # Index XY Length
    xyLink87 = ((x8IndexTip-x7IndexDip)**2 + (y8IndexTip-y7IndexDip)**2)**0.5 # H Knuckle
    xyLink76 = ((x7IndexDip-x6IndexPip)**2 + (y7IndexDip-y6IndexPip)**2)**0.5 # G Knuckle
    xyLink65 = ((x6IndexPip-x5IndexMcp)**2 + (y6IndexPip-y5IndexMcp)**2)**0.5 # F Knuckle

    #Arbitrary Lengths
    Length54 = ((x4ThumbTip-x5IndexMcp)**2 + (y4ThumbTip-y5IndexMcp)**2 + (z4ThumbTip-z5IndexMcp)**2)**0.5  
    Length84 = ((x8IndexTip-x4ThumbTip)**2 + (y8IndexTip-y4ThumbTip)**2 + (z8IndexTip-z4ThumbTip)**2)**0.5


    totLinkLength85 = Link87 + Link76 + Link65
    xLength85 = x8IndexTip - x5IndexMcp
    yLength85 = y8IndexTip - y5IndexMcp
    zLength85 = z8IndexTip - z5IndexMcp
    xyLength85 =  (xLength85**2 + yLength85**2)**0.5
    yzLength85 =  (yLength85**2 + zLength85**2)**0.5
    xzLength85 =  (xLength85**2 + zLength85**2)**0.5
    xyzLength85 = (xLength85**2 + yLength85**2 + zLength85**2)**0.5
    xyzLength85COMP = (xLength85**2 + yLength85**2 + 12*zLength85**2)**0.5 #Compensate for z inacc. ~ 3 --> sqrt12

    Ratio = xyLength85/totLinkLength85

    # Roll, 8 = pointer, 17 = pivot, y+down
    if x5IndexMcp >= x17PinkyMcp and y5IndexMcp >= y17PinkyMcp: # Quadrant 1
        rollAngle = math.degrees(math.atan((y5IndexMcp-y17PinkyMcp)/(x5IndexMcp-x17PinkyMcp)))
    elif x5IndexMcp < x17PinkyMcp and y5IndexMcp >= y17PinkyMcp: # Quadrant 2
        rollAngle = math.degrees(math.atan((y5IndexMcp-y17PinkyMcp)/(x5IndexMcp-x17PinkyMcp))) + 180
    elif x5IndexMcp < x17PinkyMcp and y5IndexMcp < y17PinkyMcp: # Quadrant 3
        rollAngle = math.degrees(math.atan((y5IndexMcp-y17PinkyMcp)/(x5IndexMcp-x17PinkyMcp))) + 180
    else:
        rollAngle = math.degrees(math.atan((y5IndexMcp-y17PinkyMcp)/(x5IndexMcp-x17PinkyMcp))) + 360
    # Pitch, y+down
    pitchAngle = math.degrees(math.asin(yLength85/xyzLength85))
    # Yaw
    yawAngle = math.degrees(math.asin(xLength85/xyzLength85))

    # Finger Gun
    #if Link43*2 <= Length54: # Trigger Motion
        #return ("Nozzle off", round(rollAngle, 2), round(pitchAngle, 2), round(yawAngle, 2))
    #else:
        #return ("Nozzle on", round(rollAngle, 2), round(pitchAngle, 2), round(yawAngle, 2))
    
    # Pinching Action
    if Link43 <= Length84: 
        return ("Nozzle off", round(rollAngle, 2), round(pitchAngle, 2), round(yawAngle, 2))
    else:
        return ("Nozzle on", round(rollAngle, 2), round(pitchAngle, 2), round(yawAngle, 2))

def main():
    # Initialize VideoStream
    video_stream = VideoStream(0)

    # Initialize MediaPipe Hands
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode       = False,
        max_num_hands           = maxNumHands,
        min_detection_confidence= minDetectionConfidence,
        min_tracking_confidence = MIN_TRACKING_CONFIDENCE,
        model_complexity        = MODEL_COMPLEXITY
    )
    mpDrawing = mp.solutions.drawing_utils

    # Variables for FPS calculation
    prev_time = time.time()
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        success, frame = video_stream.read()
        if not success or frame is None:
            break

        # Optional: Frame Skipping
        # Implement frame skipping logic here if frameSkip > 0

        # Preprocessing
        frame = cv2.flip(frame, 1)  # Flip horizontally
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and find hands using MediaPipe
        results = hands.process(rgb_frame)

        # Post-processing
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        # Overlay FPS on the frame
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

                print(pointerData(handLandmarks))

        # Display the resulting frame
        cv2.imshow('Gesture Recognition', frame)

        # Exit Mechanism: Press any key to quit
        if cv2.waitKey(1) != -1: break

    # Cleanup
    video_stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()