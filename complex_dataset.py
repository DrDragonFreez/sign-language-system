import cv2
import mediapipe as mp
import csv

# Инициализация MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
drawer = mp.solutions.drawing_utils  # Для отрисовки

# Камера
cap = cv2.VideoCapture(0)

# Название жеста
label = input("Введите название сложного жеста: ")

samples = []
recording = False

print("Нажми 'r' чтобы начать запись, 's' чтобы остановить, 'q' для выхода")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        recording = True
        print("Запись началась")
    elif key == ord('s'):
        recording = False
        print("Запись остановлена")
    elif key == ord('q'):
        break

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            drawer.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)  #отрисовка

    if recording and result.multi_hand_landmarks:
        row = []
        coords = []
        hands_landmarks = result.multi_hand_landmarks

        for i in range(2):
            if i < len(hands_landmarks):
                hand = hands_landmarks[i]
                base_x = hand.landmark[0].x
                base_y = hand.landmark[0].y
                base_z = hand.landmark[0].z

                for lm in hand.landmark:
                    coords.append(lm.x - base_x)
                    coords.append(lm.y - base_y)
                    coords.append(lm.z - base_z)
            else:
                coords.extend([0.0] * 63)

        row.extend(coords)
        row.append(label)
        samples.append(row)

    # UI надпись
    cv2.putText(frame, f"Сохранил: {len(samples)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Показ окна
    cv2.imshow("Сбор сложных жестов", frame)

    if cv2.getWindowProperty("Сбор сложных жестов", cv2.WND_PROP_VISIBLE) < 1:
        break

# Завершение
cap.release()
cv2.destroyAllWindows()

# Сохранение
with open("dataset.csv", "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(samples)

print(f"Сохранили {len(samples)} записей в dataset.csv")
