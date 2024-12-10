import cv2
import mediapipe as mp
import math
import socket
import struct

# Calibration for 3D depth and distance calculation
FINGER_DISTANCE_CM = 10.5

# Function to calculate 3D distance
def calculate_3d_distance(point1, point2):
    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2 +
        (point1.z - point2.z) ** 2
    )

# Initialize socket server
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 65432
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print("Waiting for Raspberry Pi to connect...")
conn, addr = server_socket.accept()
print(f"Connected to {addr}")

# Initialize MediaPipe Hands and Face Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        image_height, image_width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 2:
            hand_1 = results.multi_hand_landmarks[0]
            hand_2 = results.multi_hand_landmarks[1]
            point1_hand1 = hand_1.landmark[8]
            point2_hand2 = hand_2.landmark[8]
            distance_norm = calculate_3d_distance(point1_hand1, point2_hand2)
            scaling_factor = FINGER_DISTANCE_CM / calculate_3d_distance(
                hand_1.landmark[8], hand_1.landmark[5]
            )
            distance_cm = distance_norm * scaling_factor

            # Send the distance to the Raspberry Pi
            conn.send(struct.pack('f', distance_cm))

        # Show video feed
        cv2.imshow('Hand Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    conn.close()
    server_socket.close()
