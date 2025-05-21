import cv2
import mediapipe as mp
import csv

# Инициализация MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

# Включаем камеру
cap = cv2.VideoCapture(0)

# Запрос названия жеста у пользователя
label = input("Введите название сложного жеста: ")

samples = []
recording = False

print("Нажми 'r' чтобы начать запись, 's' чтобы остановить, 'q' для выхода")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Отзеркаливаем изображение
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Обрабатываем изображение
    result = hands.process(rgb)

    # Обработка клавиш
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        recording = True
        print("▶️ Запись началась")

    if key == ord('s'):
        recording = False
        print("⏹️ Запись остановлена")

    if key == ord('q'):
        break

    # Если запись включена и есть обнаруженные руки
    if recording and result.multi_hand_landmarks:
        row = []
        coords = []

        hands_landmarks = result.multi_hand_landmarks

        # Обрабатываем максимум 2 руки
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
                # Заполняем нулями, если вторая рука отсутствует
                coords.extend([0.0] * 63)

        row.extend(coords)
        row.append(label)
        samples.append(row)

    # Отображаем количество сохранённых записей
    cv2.putText(
        frame,
        f"Сохранил: {len(samples)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Показываем окно
    cv2.imshow("Сбор сложных жестов", frame)

    # Завершаем при закрытии окна
    if cv2.getWindowProperty("Сбор сложных жестов", cv2.WND_PROP_VISIBLE) < 1:
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()

# Сохраняем жесты в CSV с поддержкой русских символов
with open("complex_dataset.csv", "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(samples)

# Итоговый вывод
if recording:
    print("⚙️ Запись всё ещё была активна на момент завершения")

    if result.multi_hand_landmarks:
        print(" Рука была обнаружена")
    else:
        print(" Рука не была обнаружена")

print(f" Сохранили {len(samples)} записей в complex_dataset.csv")