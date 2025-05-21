import cv2
import mediapipe as mp
from recognizer import extract_landmark_vector, predict_gesture

# Инициализируем распознавание рук через MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Распознаём до двух рук
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Подключаем камеру
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Отзеркаливаем изображение, чтобы было как в зеркале
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Получаем результат обработки кадра
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Рисуем соединения между точками руки
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Преобразуем координаты точек в вектор признаков
            feature_vector = extract_landmark_vector(hand_landmarks)

            # Распознаём жест
            gesture = predict_gesture(feature_vector)

            # Показываем название распознанного жеста на экране
            cv2.putText(
                frame,
                f'Жест: {gesture}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    # Отображаем окно с результатом
    cv2.imshow('Распознавание жестов (2 руки)', frame)

    # Выход по 'q' или закрытию окна
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty('Распознавание жестов (2 руки)', cv2.WND_PROP_VISIBLE) < 1:
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()