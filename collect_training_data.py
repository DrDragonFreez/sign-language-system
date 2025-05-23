import cv2
import os
import mediapipe as mp
import csv
import numpy as np

TEMPLATE_DIR = "templates"
OUTPUT_CSV = "gesture_dataset.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

header = []
for i in range(21):
    header += [f"xL{i}", f"yL{i}", f"zL{i}"]
for i in range(21):
    header += [f"xR{i}", f"yR{i}", f"zR{i}"]
header += ["label"]

with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for category in os.listdir(TEMPLATE_DIR):
        category_path = os.path.join(TEMPLATE_DIR, category)
        if os.path.isdir(category_path):
            for file in os.listdir(category_path):
                if not file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    continue

                label = os.path.splitext(file)[0]
                file_path = os.path.join(category_path, file)
                cap = cv2.VideoCapture(file_path)

                if not cap.isOpened():
                    print(f"Не удалось открыть {file_path}")
                    continue

                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(frame_rgb)

                    left_hand = [0.0] * 63
                    right_hand = [0.0] * 63

                    if results.multi_hand_landmarks and results.multi_handedness:
                        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                            handedness = results.multi_handedness[i].classification[0].label
                            landmarks = []
                            for lm in hand_landmarks.landmark:
                                landmarks.extend([lm.x, lm.y, lm.z])

                            if handedness == 'Left':
                                left_hand = landmarks
                            else:
                                right_hand = landmarks

                    row = left_hand + right_hand + [label]
                    writer.writerow(row)

                    frame_count += 1
                    if frame_count >= 3000:
                        break

                cap.release()

print("Извлечение признаков завершено. Данные сохранены в gesture_dataset.csv")
