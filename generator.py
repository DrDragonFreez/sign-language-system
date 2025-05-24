import cv2
import os
import json
from PIL import ImageFont, ImageDraw, Image
import numpy as np

# Путь к словарю с расшифровкой жестов
DICT_PATH = "dictionary.json"
# Папка с видеофайлами
TEMPLATE_DIR = "templates"
# Путь к шрифту с поддержкой кириллицы
FONT_PATH = "arial.ttf"
FONT_SIZE = 32
TEXT_COLOR = (0, 255, 0)

# Загрузка словаря сопоставления
with open(DICT_PATH, "r", encoding="utf-8") as f:
    gesture_dict = json.load(f)

# Сканируем подпапки: буквы, цифры, слова
gesture_videos = []
for category in os.listdir(TEMPLATE_DIR):
    category_path = os.path.join(TEMPLATE_DIR, category)
    if os.path.isdir(category_path):
        for file in os.listdir(category_path):
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                gesture_name = os.path.splitext(file)[0]  # без расширения
                full_path = os.path.join(category_path, file)
                description = gesture_dict.get(gesture_name, gesture_name)
                gesture_videos.append((gesture_name, description, full_path))

# Выводим список доступных жестов
print("Доступные жесты:")
for i, (name, desc, _) in enumerate(gesture_videos):
    print(f"[{i}] {name} — {desc}")

# Запрос у пользователя
choice = input("Введите номер жеста для воспроизведения: ")
if not choice.isdigit() or int(choice) >= len(gesture_videos):
    print("Неверный выбор.")
    exit(1)

_, description, video_path = gesture_videos[int(choice)]

# Функция для отображения текста с кириллицей

def draw_text_with_pil(frame, text, position):
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except IOError:
        print("Не найден файл шрифта arial.ttf. Поместите его рядом со скриптом.")
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=TEXT_COLOR)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# Открытие видеофайла и воспроизведение
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Не удалось открыть видео.")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Отображение текста (название жеста)
    frame = draw_text_with_pil(frame, description, (20, 30))
    cv2.imshow("Воспроизведение жеста", frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q') or cv2.getWindowProperty("Воспроизведение жеста", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
