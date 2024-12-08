import cv2
import mediapipe as mp
import math

FINGER_DISTANCE_CM = 10.5

def calculate_3d_distance(point1, point2):
    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2 +
        (point1.z - point2.z) ** 2
   )

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    image_height, image_width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face = face_detection.process(frame_rgb)

    if results_face.detections:
        for detection in results_face.detections:
            # Draw the face detection annotations
            mp_drawing.draw_detection(frame, detection)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if len(results.multi_hand_landmarks) >= 2:
            hand_1 = results.multi_hand_landmarks[0]
            hand_2 = results.multi_hand_landmarks[1]
            point1 = hand_1.landmark[8]
            point2 = hand_1.landmark[5]
            finger_distance_norm = calculate_3d_distance(point1, point2)
            scaling_factor = FINGER_DISTANCE_CM / finger_distance_norm
            point1_hand1 = hand_1.landmark[8]
            point2_hand2 = hand_2.landmark[8]
            distance_norm = calculate_3d_distance(point1_hand1, point2_hand2)
            distance_cm = distance_norm * scaling_factor

            x1, y1 = int(point1_hand1.x * image_width), int(point1_hand1.y * image_height)
            x2, y2 = int(point2_hand2.x * image_width), int(point2_hand2.y * image_height)

            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            cv2.putText(
                frame,
                f"{distance_cm:.2f} cm",
                (mid_x, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.imshow('Hand Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


#FACE DETECTION

