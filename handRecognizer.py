import cv2
import mediapipe as mp           #imported libraries
import math

FINGER_DISTANCE_CM = 10.5      #calibration for 3d deptyh and distance calc

def calculate_3d_distance(point1, point2):
    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2 +             #copyied function to calculate distance
        (point1.z - point2.z) ** 2
   )

camera_index = 1      #to choose which camera to use

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils   #hand recognition
cap = cv2.VideoCapture(camera_index)

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils #face detection

while cap.isOpened():
    ret, frame = cap.read()   #reading video feed
    if not ret:
        continue

    image_height, image_width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)   #hand recognition

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face = face_detection.process(frame_rgb) #face recognition

    if results_face.detections:
        for detection in results_face.detections:
            mp_drawing.draw_detection(frame, detection) #draw face detection

    if results.multi_hand_landmarks:      #if 2 or more hands detected
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if len(results.multi_hand_landmarks) >= 2:
            hand_1 = results.multi_hand_landmarks[0]
            hand_2 = results.multi_hand_landmarks[1]
            point1 = hand_1.landmark[8]     #finger landmarks accorig to mediapipe
            point2 = hand_1.landmark[5]
            finger_distance_norm = calculate_3d_distance(point1, point2)
            scaling_factor = FINGER_DISTANCE_CM / finger_distance_norm
            point1_hand1 = hand_1.landmark[8]
            point2_hand2 = hand_2.landmark[8]
            distance_norm = calculate_3d_distance(point1_hand1, point2_hand2)
            distance_cm = distance_norm * scaling_factor
            #covert to pixel distance to calculate
            x1, y1 = int(point1_hand1.x * image_width), int(point1_hand1.y * image_height)
            x2, y2 = int(point2_hand2.x * image_width), int(point2_hand2.y * image_height)

            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            cv2.putText(
                frame,
                f"{distance_cm:.2f} cm",     #putting distance and line on screen
                (mid_x, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.imshow('Hand Recognition', frame)   #showing the video feed

    if cv2.waitKey(1) & 0xFF == ord('q'):   #press q to exit
        break

cap.release()
cv2.destroyAllWindows()    #closing the video feed





