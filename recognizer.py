import numpy as np
import joblib

try:
    model = joblib.load("model.pkl")
except FileNotFoundError:
    print("[!] Внимание: model.pkl не найден. Классификация работать не будет.")
    model = None


def extract_landmark_vector(multi_hand_landmarks):
    def normalize(landmarks):
        base_x = landmarks[0].x
        base_y = landmarks[0].y
        base_z = landmarks[0].z
        return [(lm.x - base_x, lm.y - base_y, lm.z - base_z) for lm in landmarks]

    vector = []

    if multi_hand_landmarks is None:
        return np.zeros(126)

    hands = list(multi_hand_landmarks)
    hands = hands[:2]

    for hand in hands:
        norm = normalize(hand.landmark)
        for x, y, z in norm:
            vector.extend([x, y, z])

    if len(hands) == 1:
        vector.extend([0.0] * 63)

    return np.array(vector)


def predict_gesture(feature_vector, threshold=0.7):
    if model is None:
        return "Не обучено"

    proba = model.predict_proba([feature_vector])[0]

    max_prob = max(proba)
    predicted_class = model.classes_[proba.argmax()]

    if max_prob < threshold:
        return None

    return predicted_class
