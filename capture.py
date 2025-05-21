import cv2
import mediapipe as mp
from recognizer import extract_landmark_vector, predict_gesture
from PIL import ImageFont, ImageDraw, Image
import numpy as np

def put_cyrillic_text(img, text, position, font_path='arial.ttf', font_size=32, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
drawing_utils = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        feature_vector = extract_landmark_vector(results.multi_hand_landmarks)
        gesture = predict_gesture(feature_vector)

        # Используем нашу функцию для вывода кириллицы
        frame = put_cyrillic_text(frame, f'Жест: {gesture}', (10, 40), font_size=40)

    cv2.imshow('Распознавание жестов (2 руки)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty('Распознавание жестов (2 руки)', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()