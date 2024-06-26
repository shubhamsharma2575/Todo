import cv2
import mediapipe as mp
import numpy as np

#Initialize MediaPipe Hands and Drawing Utilities

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# For webcam input:
cap = cv2.VideoCapture(0)

def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2 + (point1.z - point2.z) ** 2)

def detect_thumbs_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    thumb_index_dist = calculate_distance(thumb_tip, index_tip)
    thumb_middle_dist = calculate_distance(thumb_tip, middle_tip)
    thumb_ring_dist = calculate_distance(thumb_tip, ring_tip)
    thumb_pinky_dist = calculate_distance(thumb_tip, pinky_tip)
    index_middle_dist = calculate_distance(index_tip,middle_tip)
    middle_ring_dist = calculate_distance(middle_tip,ring_tip)
    ring_pinky_dist = calculate_distance(ring_tip,pinky_tip)
    ring_palm_dist = calculate_distance(ring_tip, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP])
    if (thumb_tip.y < thumb_ip.y and 
        thumb_index_dist > 0.1 and 
        thumb_middle_dist > 0.23 and 
        thumb_ring_dist > 0.23 and 
        thumb_pinky_dist > 0.23 and
        index_middle_dist < 0.06 and
        middle_ring_dist < 0.06 and
        ring_pinky_dist < 0.06 and
        ring_palm_dist < 0.06):
        return True
    return False

def detect_thumbs_down(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    thumb_index_dist = calculate_distance(thumb_tip, index_tip)
    thumb_middle_dist = calculate_distance(thumb_tip, middle_tip)
    thumb_ring_dist = calculate_distance(thumb_tip, ring_tip)
    thumb_pinky_dist = calculate_distance(thumb_tip, pinky_tip)
    index_middle_dist = calculate_distance(index_tip,middle_tip)
    middle_ring_dist = calculate_distance(middle_tip,ring_tip)
    ring_pinky_dist = calculate_distance(ring_tip,pinky_tip)
    ring_palm_dist = calculate_distance(ring_tip, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP])

    if (thumb_tip.y > thumb_ip.y and
        thumb_index_dist > 0.1 and
        thumb_middle_dist > 0.15 and
        thumb_ring_dist > 0.15 and
        thumb_pinky_dist > 0.15 and
        index_middle_dist < 0.06 and
        middle_ring_dist < 0.06 and
        ring_pinky_dist < 0.06 and
        ring_palm_dist < 0.06):
        return True
    return False


def detect_open_hand(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    distances = [
        calculate_distance(thumb_tip, index_tip),
        calculate_distance(thumb_tip, middle_tip),
        calculate_distance(thumb_tip, ring_tip),
        calculate_distance(thumb_tip, pinky_tip),
        calculate_distance(index_tip, middle_tip),
        calculate_distance(index_tip, ring_tip),
        calculate_distance(index_tip, pinky_tip),
        calculate_distance(middle_tip, ring_tip),
        calculate_distance(middle_tip, pinky_tip),
        calculate_distance(ring_tip, pinky_tip)
    ]
    
    if all(dist > 0.06 for dist in distances):
        return True
    return False

def detect_fist(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    distances = [
        calculate_distance(thumb_tip, index_tip),
        calculate_distance(thumb_tip, middle_tip),
        calculate_distance(thumb_tip, ring_tip),
        calculate_distance(thumb_tip, pinky_tip),
        calculate_distance(index_tip, middle_tip),
        calculate_distance(index_tip, ring_tip),
        calculate_distance(index_tip, pinky_tip),
        calculate_distance(middle_tip, ring_tip),
        calculate_distance(middle_tip, pinky_tip),
        calculate_distance(ring_tip, pinky_tip)
    ]
    
    if all(dist < 0.15 for dist in distances):
        return True
    return False

def detect_peace_sign(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    index_middle_dist = calculate_distance(index_tip, middle_tip)
    index_ring_dist = calculate_distance(index_tip, ring_tip)
    middle_ring_dist = calculate_distance(middle_tip, ring_tip)
    ring_pinky_dist = calculate_distance(ring_tip, pinky_tip)
    
    if (index_middle_dist < 0.3 and 
        index_ring_dist > 0.1 and 
        middle_ring_dist > 0.1 and 
        ring_pinky_dist < 0.3):
        return True
    return False

def detect_nice(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    thumb_index_dist = calculate_distance(thumb_tip, index_tip)
    
    middle_palm_dist = calculate_distance(middle_tip, hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP])
    ring_palm_dist = calculate_distance(ring_tip, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP])
    pinky_palm_dist = calculate_distance(pinky_tip, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP])
    
    if (thumb_index_dist < 0.08 and 
        middle_palm_dist > 0.15 and 
        ring_palm_dist > 0.15 and 
        pinky_palm_dist > 0.15):
        return True
    return False

def detect_rock_and_roll(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    index_middle_dist = calculate_distance(index_tip, middle_tip)
    middle_ring_dist = calculate_distance(middle_tip, ring_tip)
    ring_pinky_dist = calculate_distance(ring_tip, pinky_tip)
    thumb_middle_dist = calculate_distance(thumb_tip, middle_tip)

    if (index_middle_dist > 0.2 and
        middle_ring_dist < 0.09 and
        ring_pinky_dist > 0.2 and
        thumb_middle_dist < 0.09):
        return True
    return False

def detect_stop(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    

    thumb_index_dist = calculate_distance(thumb_tip, index_tip)
    index_middle_dist = calculate_distance(index_tip, middle_tip)
    middle_ring_dist = calculate_distance(middle_tip, ring_tip)
    ring_pinky_dist = calculate_distance(ring_tip, pinky_tip)
    ring_palm_dist = calculate_distance(ring_tip, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP])
    
    if (thumb_index_dist < 0.23 and
        index_middle_dist < 0.09 and
        middle_ring_dist < 0.09 and
        ring_pinky_dist < 0.09 and
        ring_palm_dist >0.08):
        return True
    return False

def detect_i_love_you(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    thumb_index_dist = calculate_distance(thumb_tip, index_tip)
    index_middle_dist = calculate_distance(index_tip, middle_tip)
    middle_ring_dist = calculate_distance(middle_tip, ring_tip)
    ring_pinky_dist = calculate_distance(ring_tip, pinky_tip)

    if (thumb_index_dist > 0.2 and
        index_middle_dist > 0.2 and
        middle_ring_dist < 0.09 and
        ring_pinky_dist > 0.2):
        return True
    return False


def detect_smile(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    thumb_index_dist = calculate_distance(thumb_tip, index_tip)
    
    middle_palm_dist = calculate_distance(middle_tip, hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP])
    ring_palm_dist = calculate_distance(ring_tip, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP])
    pinky_palm_dist = calculate_distance(pinky_tip, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP])
    index_middle_dist = calculate_distance(index_tip, middle_tip)
   
    
    if (thumb_index_dist > 0.15 and 
        middle_palm_dist < 0.05 and 
        ring_palm_dist < 0.05 and 
        pinky_palm_dist < 0.05 and
        index_middle_dist > 0.1):
        return True
    return False

def detect_call_me(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]

    thumb_pinky_dist = calculate_distance(thumb_tip, pinky_tip)
    index_middle_dist = calculate_distance(index_tip, middle_tip)
    middle_ring_dist = calculate_distance(middle_tip, ring_tip)
    ring_palm_dist = calculate_distance(ring_tip, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP])
    pinky_palm_dist = calculate_distance(pinky_tip, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP])

    if (thumb_pinky_dist > 0.3 and
        index_middle_dist < 0.09 and
        middle_ring_dist < 0.09 and
        ring_palm_dist < 0.09 and
        pinky_palm_dist > 0.1):
        return True
    return False


# Integration into the main loop
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        label = ""
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                if detect_thumbs_up(hand_landmarks):
                    label = "OK"
                elif detect_open_hand(hand_landmarks):
                    label = "Hello"
                elif detect_fist(hand_landmarks):
                    label = "YES"
                elif detect_peace_sign(hand_landmarks):
                    label = "Peace Sign"
                elif detect_nice(hand_landmarks):
                    label = "Nice"    
                elif detect_rock_and_roll(hand_landmarks):
                    label = "ROCK and ROLL" 
                elif detect_smile(hand_landmarks):
                    label = "Smile"
                elif detect_i_love_you(hand_landmarks):
                    label = "I Love You"        
                elif detect_call_me(hand_landmarks):
                    label = "Call Me"
                elif detect_thumbs_down(hand_landmarks):
                    label = "No"  
                elif detect_stop(hand_landmarks):
                    label = "Stop"     
                    
        if label:
            cv2.putText(image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
