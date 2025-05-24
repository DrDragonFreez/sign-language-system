import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import os
import json
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
from collections import deque, Counter
from recognizer import extract_landmark_vector as extract_static_vector, predict_gesture
from recognizer1 import extract_landmark_vector as extract_dynamic_vector, predict_dynamic


def draw_text(img, text, pos, font_path='arial.ttf', size=32, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, size)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# Загружаем словарь жестов
try:
    with open("dictionary.json", "r", encoding="utf-8") as f:
        gesture_dict = json.load(f)
except FileNotFoundError:
    gesture_dict = {}
    print("dictionary.json не найден — будет использоваться прямой поиск по имени.")


class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Жестовый Переводчик")
        self.running = True

        # Камера/видео окно
        self.video_label = ttk.Label(root)
        self.video_label.pack()

        # Распознанный текст
        ttk.Label(root, text="Распознанный текст:").pack()
        self.text_var = tk.StringVar()
        self.text_label = ttk.Label(root, textvariable=self.text_var, font=("Arial", 14))
        self.text_label.pack()

        # Поле ввода
        ttk.Label(root, text="Введите слово:").pack()
        self.input_entry = ttk.Entry(root, width=30)
        self.input_entry.pack()

        # Кнопки
        btns = ttk.Frame(root)
        btns.pack(pady=10)

        ttk.Button(btns, text="Показать жесты", command=self.show_gesture).grid(row=0, column=0, padx=5)
        ttk.Button(btns, text="Сброс", command=self.clear_text).grid(row=0, column=1, padx=5)
        ttk.Button(btns, text="Переключить режим", command=self.toggle_mode).grid(row=0, column=2, padx=5)
        ttk.Button(btns, text="Стоп", command=self.stop).grid(row=0, column=3, padx=5)

        # Камера и распознавание
        self.cap = cv2.VideoCapture(0)
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.drawing = mp.solutions.drawing_utils

        # Состояния
        self.frame_count = 0
        self.mode = 'static'
        self.history = deque(maxlen=10)
        self.dynamic_history = deque(maxlen=40)
        self.last_static = None
        self.static_timer = 0
        self.last_dynamic = None
        self.dynamic_timer = 0

        self.update()

    def update(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        out = frame.copy()
        self.frame_count += 1

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                self.drawing.draw_landmarks(out, hand, mp.solutions.hands.HAND_CONNECTIONS)

            if self.mode == 'static' and self.frame_count % 3 == 0:
                vec = extract_static_vector(results.multi_hand_landmarks)
                pred = predict_gesture(vec)
                self.history.append(pred)
                if self.history:
                    common = Counter(self.history).most_common(1)[0][0]
                    if common != self.last_static:
                        self.last_static = common
                        self.static_timer = 30

            elif self.mode == 'dynamic' and self.frame_count % 3 == 0:
                vec = extract_dynamic_vector(results.multi_hand_landmarks)
                self.dynamic_history.append(vec)
                if len(self.dynamic_history) == 40:
                    pred = predict_dynamic(list(self.dynamic_history))
                    if pred:
                        self.last_dynamic = pred
                        self.dynamic_timer = 30
                        self.dynamic_history.clear()

        if self.mode == 'static' and self.static_timer > 0 and self.last_static:
            self.text_var.set(self.last_static)
            self.static_timer -= 1

        if self.mode == 'dynamic' and self.dynamic_timer > 0 and self.last_dynamic:
            self.text_var.set(self.last_dynamic)
            self.dynamic_timer -= 1

        # Подпись режима
        if self.mode == "static":
            out = draw_text(out, "Режим: СТАТИЧЕСКИЙ", (10, 10), size=28, color=(0, 255, 0))
            color = (0, 255, 0)
        else:
            out = draw_text(out, "Режим: ДИНАМИЧЕСКИЙ", (10, 10), size=28, color=(0, 128, 255))
            color = (0, 128, 255)

        out = cv2.copyMakeBorder(out, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=color)

        img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        self.root.after(30, self.update)

    def show_gesture(self):
        word = self.input_entry.get().strip().lower()
        if not word:
            self.text_var.set("Введите слово для показа")
            return

        filename = gesture_dict.get(word, f"{word}.mp4")
        path = os.path.join("templates", filename)

        if not os.path.exists(path):
            self.text_var.set(f"Файл не найден: {filename}")
            return

        self.text_var.set(f"Показ: {word}")
        self.running = False

        cap = cv2.VideoCapture(path)

        def play():
            ret, frame = cap.read()
            if not ret:
                cap.release()
                self.running = True
                self.update()
                return

            frame = cv2.resize(frame, (640, 480))
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

            self.root.after(30, play)

        play()

    def toggle_mode(self):
        self.mode = 'dynamic' if self.mode == 'static' else 'static'
        self.last_static = None
        self.static_timer = 0
        self.last_dynamic = None
        self.dynamic_timer = 0
        self.dynamic_history.clear()

    def clear_text(self):
        self.input_entry.delete(0, tk.END)
        self.text_var.set("")

    def stop(self):
        self.running = False
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = GestureApp(root)
    root.mainloop()
