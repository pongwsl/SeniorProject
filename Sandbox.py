#
# sandbox.py
# only for testing.
#

# --- FUNCTION ---
import math

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



def recognizeGesture(lm):    # landmarks
    '''This function recognizes hand gestures using hand landmarks.'''
    fingerDowns = [0, 0, 0, 0, 0]

    # finger down recognition
    # not thumb
    for i in range(1, 5):
        fingerDowns[i] = ((((lm.landmark[0].y > lm.landmark[4*i+4].y > lm.landmark[10].y) or
                            (lm.landmark[0].y < lm.landmark[4*i+4].y < lm.landmark[10].y)) and
                           ((lm.landmark[2].x > lm.landmark[4*i+4].x > lm.landmark[17].x) or
                            (lm.landmark[2].x < lm.landmark[4*i+4].x < lm.landmark[17].x))) or
                          (((lm.landmark[0].x > lm.landmark[4*i+4].x > lm.landmark[10].x) or
                            (lm.landmark[0].x < lm.landmark[4*i+4].x < lm.landmark[10].x)) and
                           ((lm.landmark[2].y > lm.landmark[4*i+4].y > lm.landmark[17].y) or
                            (lm.landmark[2].y < lm.landmark[4*i+4].y < lm.landmark[17].y))))
    # Thumb finger
    fingerDowns[0] = (((lm.landmark[0].x > lm.landmark[1].x) != 
                       (lm.landmark[3].x > lm.landmark[4].x)) or 
                      ((lm.landmark[0].y > lm.landmark[1].y) != 
                       (lm.landmark[3].y > lm.landmark[4].y)))

    # Sign
    fist        = fingerDowns == [1, 1, 1, 1, 1]
    openHand    = fingerDowns == [0, 0, 0, 0, 0]
    peaceSign   = fingerDowns == [1, 0, 0, 1, 1]
    rockSign    = fingerDowns == [1, 0, 1, 1, 0]
    ilySign     = fingerDowns == [0, 0, 1, 1, 0]
    helloSign   = fingerDowns == [1, 1, 0, 1, 1]
    KawaiSign   = fingerDowns == [0, 1, 1, 1, 1]
    GermanThree = fingerDowns == [0, 0, 0, 1, 1]
    EnglishThree = fingerDowns == [1, 0, 0, 0, 1]
    Pointing = fingerDowns == [1, 0, 1, 1, 1]

    # Index finger
    indexFinger = fingerDowns == [1, 0, 1, 1, 1]
    if indexFinger:
        tangent  = math.degrees(math.atan2(lm.landmark[5].y - lm.landmark[8].y,
                                           lm.landmark[8].x - lm.landmark[5].x))
        if abs(tangent) > 120:
            return 'Index Finger Left'
        elif tangent > 60:
            return 'Index Finger Up'
        elif tangent < -60:
            return 'Index Finger Down'
        else:
            return 'Index Finger Right'

    # Return
    if fist:
        return 'Fist'
    elif openHand:
        return 'Open Hand'
    elif peaceSign:
        return 'Peace Sign'
    elif rockSign:
        return 'Rock Sign'
    elif ilySign:
        return 'I Love You Sign'
    elif helloSign:
        return '>:['
    elif KawaiSign:
        return 'Kawai Sign'
    elif GermanThree:
        return 'Fruend:)'
    elif EnglishThree:
        return 'Enemy of the state'
    elif Pointing:
        return 'Plotting'
    else:
        return f'Unknown Gesture: {fingerDowns}'
    
# --- MAIN ---
def main():
    import cv2
    import mediapipe as mp

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    mpDrawing = mp.solutions.drawing_utils
    camera = cv2.VideoCapture(0)

    while camera.isOpened():
        success, frame = camera.read()
        if not success: break

        # Determine the smaller dimension to crop a square frame
        height, width, _ = frame.shape
        size = min(height, width)

        # Calculate cropping region to center the frame
        start_x = (width - size) // 2
        start_y = (height - size) // 2
        frame = frame[start_y:start_y+size, start_x:start_x+size]

        # Flip the image horizontally for a later selfie-view display and also convert the color space from BGR to RGB
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        
        # Process the frame and find hands using MediaPipe
        results = hands.process(frame)

        # Draw the hand annotations on the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks:
                mpDrawing.draw_landmarks(frame, handLandmarks, mpHands.HAND_CONNECTIONS)
                gestureText = recognizeGesture(handLandmarks)
                tipCoordText = tipCoord(handLandmarks,1)
                triggerText = pointerData(handLandmarks)
                
                cv2.putText(frame,  # The image we wanted to put text on
                            f'{tipCoordText}',   # Text
                            (10, 80),   # Position in (x, y)
                            cv2.FONT_HERSHEY_DUPLEX,   # Font
                            0.7,  # FontScale
                            (0, 0, 0),  # Color
                            2)  # FontThickness
                
                cv2.putText(frame,  # The image we wanted to put text on
                            f'{triggerText}',   # Text
                            (10, 140),   # Position in (x, y)
                            cv2.FONT_HERSHEY_DUPLEX,   # Font
                            0.7,  # FontScale
                            (0, 0, 0),  # Color
                            2)  # FontThickness
                
        cv2.imshow('Gesture Recognition', frame)  # Display the frame
        key = cv2.waitKey(1)
        if key != -1: break # Press any key to quit

    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()