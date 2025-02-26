from src.config import CLASSES
from src.model import QuickDraw
import torch
import numpy as np
from src.utils import preprocess_image, get_random_class
import cv2
import mediapipe as mp
import time

def main():
    # Initialize predicted_class_name
    predicted_class_name = None
    last_trigger_time = 0
    debounce_time = 1  # 1 second debounce time

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = QuickDraw(num_classes=len(CLASSES))
    load_model = torch.load("quickdraw_model.pth", map_location=torch.device('cpu'))
    model.load_state_dict(load_model)
    model.to(device)

    while True:
        # Randomly select a class before drawing
        random_class = get_random_class()
        print(f"Randomly selected class: {random_class}")

        # ตั้งค่า Mediapipe
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

        # ขนาดเฟรม
        width, height = 640, 480
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        drawing = False
        prev_point = None

        # ตำแหน่งของโซน Save, Clear และ Get Random Class
        save_zone = (width - 120, 20, width - 20, 100)
        clear_zone = (20, 20, 120, 100)
        random_zone = (width // 2 - 60, 20, width // 2 + 60, 100)

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
                        current_time = time.time()
                        if current_time - last_trigger_time > debounce_time:
                            if save_zone[0] <= middle_finger_tip[0] <= save_zone[2] and save_zone[1] <= middle_finger_tip[1] <= save_zone[3]:
                                cv2.imwrite("drawing.png", canvas)
                                print("Saved drawing.png")

                                # Load the saved drawing
                                image_path = "drawing.png"
                                input_tensor = preprocess_image(image_path)

                                # Move the input tensor to the same device as the model
                                input_tensor = input_tensor.to(device)

                                # Perform inference
                                with torch.no_grad():
                                    logits = model(input_tensor)

                                # Get the predicted class
                                predicted_class = torch.argmax(logits[0])

                                # Print the predicted class
                                predicted_class_name = CLASSES[predicted_class]
                                print(f"Predicted class: {predicted_class_name}")

                                last_trigger_time = current_time

                            elif clear_zone[0] <= middle_finger_tip[0] <= clear_zone[2] and clear_zone[1] <= middle_finger_tip[1] <= clear_zone[3]:
                                canvas[:] = 0
                                print("Canvas cleared")
                                last_trigger_time = current_time

                            elif random_zone[0] <= middle_finger_tip[0] <= random_zone[2] and random_zone[1] <= middle_finger_tip[1] <= random_zone[3]:
                                random_class = get_random_class()
                                print(f"New randomly selected class: {random_class}")
                                canvas[:] = 0  # Clear the canvas
                                predicted_class_name = None  # Reset the predicted class name
                                last_trigger_time = current_time
                    
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
            
            # ui
            cv2.rectangle(frame, save_zone[:2], save_zone[2:], (0, 200, 255), -1)
            cv2.putText(frame, "Save", (save_zone[0] + 10, save_zone[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.rectangle(frame, clear_zone[:2], clear_zone[2:], (255, 100, 100), -1)
            cv2.putText(frame, "Clear", (clear_zone[0] + 10, clear_zone[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.rectangle(frame, random_zone[:2], random_zone[2:], (100, 255, 100), -1)
            cv2.putText(frame, "Random", (random_zone[0] + 10, random_zone[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Display random class and predicted class name
            cv2.putText(frame, f"Random Class: {random_class}", (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Predicted Class: {predicted_class_name}", (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Hand Drawing", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        # Ask the user if they want to draw another class
        user_input = input("Do you want to draw another class? (y/n): ")
        if user_input.lower() != 'y':
            break

if __name__ == "__main__":
    main()