import cv2
import mediapipe as mp
from recognizer import extract_landmark_vector as extract_static_vector, predict_gesture
from recognizer1 import extract_landmark_vector as extract_dynamic_vector, predict_dynamic
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from collections import deque, Counter

def draw_cyrillic_text(image, text, position, font_path='arial.ttf', font_size=32, color=(0, 255, 0)):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# Параметры
DYNAMIC_SEQUENCE_LENGTH = 10  # нужно меньше кадров
STATIC_DISPLAY_TIME = 30

# MediaPipe Hands
hands_module = mp.solutions.hands
hands_detector = hands_module.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
drawing_utils = mp.solutions.drawing_utils

# Камера
camera = cv2.VideoCapture(0)

# Истории
history = deque(maxlen=10)
dynamic_history = deque(maxlen=DYNAMIC_SEQUENCE_LENGTH)
frame_count = 0

# Предсказания
last_dynamic_prediction = None
dynamic_display_counter = 0
last_static_prediction = None
static_display_counter = 0

# Режим
mode = 'static'

while True:
    success, frame = camera.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb_frame)

    display_frame = frame.copy()
    frame_count += 1

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            drawing_utils.draw_landmarks(display_frame, hand_landmarks, hands_module.HAND_CONNECTIONS)

        if mode == 'static' and frame_count % 3 == 0:
            vector_static = extract_static_vector(results.multi_hand_landmarks)
            prediction = predict_gesture(vector_static)
            history.append(prediction)

            if history:
                most_common = Counter(history).most_common(1)[0][0]
                if most_common != last_static_prediction:
                    last_static_prediction = most_common
                    static_display_counter = STATIC_DISPLAY_TIME

        if mode == 'static' and static_display_counter > 0 and last_static_prediction:
            display_frame = draw_cyrillic_text(display_frame, f'Жест: {last_static_prediction}', (10, 40), font_size=40)
            static_display_counter -= 1

        elif mode == 'dynamic' and frame_count % 3 == 0:
            vector_dynamic = extract_dynamic_vector(results.multi_hand_landmarks)
            dynamic_history.append(vector_dynamic)

            if len(dynamic_history) == DYNAMIC_SEQUENCE_LENGTH:
                dynamic_prediction = predict_dynamic(list(dynamic_history))
                if dynamic_prediction:
                    last_dynamic_prediction = dynamic_prediction
                    dynamic_display_counter = 30
                    dynamic_history.clear()

        if mode == 'dynamic':
            display_frame = draw_cyrillic_text(display_frame, f'Считано кадров: {len(dynamic_history)}/{DYNAMIC_SEQUENCE_LENGTH}', (10, 80), font_size=28, color=(255, 255, 0))

        if dynamic_display_counter > 0 and last_dynamic_prediction:
            display_frame = draw_cyrillic_text(display_frame, f'Динамический: {last_dynamic_prediction}', (10, 120), font_size=40)
            dynamic_display_counter -= 1

    # Режим (цветовая подсветка)
    if mode == "static":
        display_frame = draw_cyrillic_text(display_frame, "Режим: СТАТИЧЕСКИЙ", (10, 10), font_size=32, color=(0, 255, 0))
        border_color = (0, 255, 0)
    else:
        display_frame = draw_cyrillic_text(display_frame, "Режим: ДИНАМИЧЕСКИЙ", (10, 10), font_size=32, color=(0, 128, 255))
        border_color = (0, 128, 255)

    # Подсказка
    display_frame = draw_cyrillic_text(display_frame, 'Нажмите M (или Ь) для смены режима, Q — выход', (10, 440), font_size=24, color=(255, 255, 255))

    # Рамка
    display_frame = cv2.copyMakeBorder(display_frame, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=border_color)

    # Показ
    cv2.imshow('Распознавание жестов', display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('m') or key == 0xFC or key == 0x044C:  # 'm' (англ) и 'ь' (рус)
        mode = 'dynamic' if mode == 'static' else 'static'
        last_static_prediction = None
        static_display_counter = 0
        last_dynamic_prediction = None
        dynamic_display_counter = 0
        dynamic_history.clear()

    if cv2.getWindowProperty('Распознавание жестов', cv2.WND_PROP_VISIBLE) < 1:
        break

camera.release()
cv2.destroyAllWindows()