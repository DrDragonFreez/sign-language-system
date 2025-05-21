import cv2
import mediapipe as mp
import csv

# Инициализируем детектор рук от MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2
)

# Открываем камеру
cap = cv2.VideoCapture(0)

# Запрашиваем у пользователя, какой жест он показывает
label = input("Введите название жеста: ")
samples = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Зеркалим изображение, чтобы было как в зеркале
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Обрабатываем кадр с помощью MediaPipe
    results = hands.process(rgb)
    row = []

    if results.multi_hand_landmarks:
        detected_hands = results.multi_hand_landmarks

        for i in range(2):  # максимум 2 руки
            if i < len(detected_hands):
                landmarks = detected_hands[i]
                base_x = landmarks.landmark[0].x
                base_y = landmarks.landmark[0].y
                base_z = landmarks.landmark[0].z

                # Сохраняем относительные координаты всех 21 точки
                for lm in landmarks.landmark:
                    row.append(lm.x - base_x)
                    row.append(lm.y - base_y)
                    row.append(lm.z - base_z)
            else:
                # Если руки меньше двух — заполняем нулями
                row.extend([0.0] * 63)

        row.append(label)
        samples.append(row)

    # Показываем количество сохранённых жестов на экране
    cv2.putText(
        frame,
        f"Сохранил: {len(samples)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Сбор жестов", frame)

    # Выход по клавише 'q' или при закрытии окна
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty("Сбор жестов", cv2.WND_PROP_VISIBLE) < 1:
        break

# Сохраняем данные в CSV с поддержкой русских символов
with open("dataset.csv", "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(samples)

print(f" Сохранили {len(samples)} записей в dataset.csv")

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()