import csv
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from recognizer import extract_landmark_vector  # та же функция, что и для статики

# константы
SEQUENCE_LEN = 40                  # кадров на один динамический жест
MAX_SAMPLES = 40                   # максимальное число примеров
DATA_FILE = "dynamic_dataset.csv"  # куда пишем
FONT = cv2.FONT_HERSHEY_SIMPLEX

# инициализация
label = input("Введите название динамического жеста: ")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
drawer = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

sequence = deque(maxlen=SEQUENCE_LEN)
samples = []
recording = False
saved_count = 0

print(" Управление:  r — запись/продолжить,  s — пауза,  q — выход")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        recording = True
    elif key == ord("s"):
        recording = False
        sequence.clear()

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            drawer.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    if recording and result.multi_hand_landmarks:
        vector = extract_landmark_vector(result.multi_hand_landmarks)
        sequence.append(vector)

        if len(sequence) == SEQUENCE_LEN:
            long_vec = np.concatenate(sequence)          # (5040,)
            samples.append(list(long_vec) + [label])
            saved_count += 1
            sequence.clear()

            if saved_count >= MAX_SAMPLES:
                print(f"Достигнуто {MAX_SAMPLES} примеров. Завершаем.")
                break

    # подписи на экране
    status_text = (
        f"Запись️  Кадров: {len(sequence)}/{SEQUENCE_LEN}"
        if recording
        else "Пауза  (r — запись)"
    )
    cv2.putText(frame, status_text, (10, 30), FONT, 0.9, (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"Примеров жеста «{label}»: {saved_count}/{MAX_SAMPLES}",
        (10, 60),
        FONT,
        0.8,
        (0, 255, 255),
        2,
    )

    cv2.imshow("Сбор динамических жестов", frame)

    if cv2.getWindowProperty("Сбор динамических жестов", cv2.WND_PROP_VISIBLE) < 1:
        break

# финал
cap.release()
cv2.destroyAllWindows()

if samples:
    with open(DATA_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(samples)

print(f"Записано {saved_count} примеров жеста «{label}» в {DATA_FILE}")
