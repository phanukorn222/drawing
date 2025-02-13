import cv2
import numpy as np
import mediapipe as mp

# ตั้งค่า Mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# ขนาดเฟรม
width, height = 640, 480
canvas = np.zeros((height, width, 3), dtype=np.uint8)
drawing = False
prev_point = None

# ตำแหน่งของโซน Save และ Clear
save_zone = (width - 120, 20, width - 20, 100)
clear_zone = (20, 20, 120, 100)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_list = hand_landmarks.landmark
            index_finger_tip = (int(landmark_list[8].x * width), int(landmark_list[8].y * height))
            middle_finger_tip = (int(landmark_list[12].x * width), int(landmark_list[12].y * height))
            
            fingers = [landmark_list[8].y < landmark_list[6].y, landmark_list[12].y < landmark_list[10].y]
            
            if fingers == [True, False]:  # ชูนิ้วชี้ วาดรูป
                drawing = True
                if prev_point is not None:
                    cv2.line(canvas, prev_point, index_finger_tip, (0, 255, 0), 5, cv2.LINE_AA)
                prev_point = index_finger_tip
            elif fingers == [True, True]:  # ชูนิ้วชี้และนิ้วกลาง หยุดวาด
                drawing = False
                prev_point = None
                
                # ตรวจสอบว่าปลายนิ้วกลางอยู่ในช่องทางไหน
                if save_zone[0] <= middle_finger_tip[0] <= save_zone[2] and save_zone[1] <= middle_finger_tip[1] <= save_zone[3]:
                    cv2.imwrite("drawing.png", canvas)
                    print("Saved drawing.png")
                elif clear_zone[0] <= middle_finger_tip[0] <= clear_zone[2] and clear_zone[1] <= middle_finger_tip[1] <= clear_zone[3]:
                    canvas[:] = 0
                    print("Canvas cleared")
            
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    
    # ui
    cv2.rectangle(frame, save_zone[:2], save_zone[2:], (0, 200, 255), -1)
    cv2.putText(frame, "Save", (save_zone[0] + 10, save_zone[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.rectangle(frame, clear_zone[:2], clear_zone[2:], (255, 100, 100), -1)
    cv2.putText(frame, "Clear", (clear_zone[0] + 10, clear_zone[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.imshow("Hand Drawing", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
