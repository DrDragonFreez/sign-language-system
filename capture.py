import cv2
import mediapipe as mp
from recognizer import extract_landmark_vector, predict_gesture
from PIL import ImageFont, ImageDraw, Image
import numpy as np

def draw_cyrillic_text(image, text, position, font_path='arial.ttf', font_size=32, color=(0, 255, 0)):
    # Преобразуем изображение в формат PIL для рисования текста
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# Инициализация детектора рук от MediaPipe
hands_module = mp.solutions.hands
hands_detector = hands_module.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
drawing_utils = mp.solutions.drawing_utils

# Запуск камеры
camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()
    if not success:
        break

    # Зеркалим изображение для эффекта зеркала
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            drawing_utils.draw_landmarks(frame, hand_landmarks, hands_module.HAND_CONNECTIONS)  # Исправлено здесь

        # Извлекаем вектор признаков и предсказываем жест
        feature_vector = extract_landmark_vector(results.multi_hand_landmarks)
        gesture = predict_gesture(feature_vector)

        # Выводим текст с предсказанным жестом
        frame = draw_cyrillic_text(frame, f'Жест: {gesture}', (10, 40), font_size=40)

    cv2.imshow('Распознавание жестов (2 руки)', frame)

    # Выход по нажатию клавиши 'q' или при закрытии окна
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty('Распознавание жестов (2 руки)', cv2.WND_PROP_VISIBLE) < 1:
        break

# Освобождаем ресурсы
camera.release()
cv2.destroyAllWindows()
