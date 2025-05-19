def extract_landmark_vector(hand_landmarks):
    """
    Преобразует landmarks в плоский вектор координат (x, y, z) с нормализацией
    """
    coords = []
    base_x = hand_landmarks.landmark[0].x
    base_y = hand_landmarks.landmark[0].y

    for lm in hand_landmarks.landmark:
        x = lm.x - base_x
        y = lm.y - base_y
        coords.append(x)
        coords.append(y)

    return coords
