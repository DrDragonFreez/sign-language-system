import numpy as np
import joblib

# Загружаем модель для динамических жестов
try:
    model = joblib.load("model1.pkl")
except FileNotFoundError:
    print("Модель model1.pkl не найдена. Распознавание недоступно.")
    model = None


def extract_landmark_vector(multi_hand_landmarks):
    """
    Извлекает нормализованные координаты x, y, z из до 2 рук.
    Возвращает 126 признаков (или нули, если рука не найдена).
    """
    def normalize(landmarks):
        base_x = landmarks[0].x
        base_y = landmarks[0].y
        base_z = landmarks[0].z
        return [(lm.x - base_x, lm.y - base_y, lm.z - base_z) for lm in landmarks]

    vector = []
    hands = list(multi_hand_landmarks)[:2] if multi_hand_landmarks else []

    for hand in hands:
        normalized = normalize(hand.landmark)
        for x, y, z in normalized:
            vector.extend([x, y, z])

    # Дополняем нулями, если вторая рука отсутствует
    while len(vector) < 126:
        vector.extend([0.0, 0.0, 0.0])

    return np.array(vector)


def predict_dynamic(sequence_40_frames, threshold=0.7):
    """
    Принимает список из 40 векторов (по 126 признаков каждый).
    Объединяет в 1 вектор длиной 5040 и предсказывает жест.
    """
    if model is None or len(sequence_40_frames) != 40:
        return None

    full_vector = np.concatenate(sequence_40_frames)  # → (5040,)
    probabilities = model.predict_proba([full_vector])[0]
    max_prob = max(probabilities)
    predicted_label = model.classes_[np.argmax(probabilities)]

    return predicted_label if max_prob >= threshold else None
