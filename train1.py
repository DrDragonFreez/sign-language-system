import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# Путь к файлу(меняем на свой куда была скачана модель)
DATA_PATH = r"C:\Users\alexh\Desktop\dynamic_dataset.csv"
MODEL_PATH = r"C:\Users\alexh\Desktop\model1.pkl"

# Загружаем вручную, проверяя длину строк
correct_rows = []
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        row = line.strip().split(',')
        if len(row) == 5041:
            correct_rows.append(row)
        else:
            print(f"Пропущена строка {i+1}: {len(row)} элементов вместо 5041")

# Преобразуем в DataFrame
df = pd.DataFrame(correct_rows)
df = df.apply(pd.to_numeric, errors='ignore')  # Преобразуем числа, а метки — оставим строками

# Разделяем на признаки и метки
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(f"Примеров: {X.shape[0]}, длина вектора: {X.shape[1]} (должно быть 5040)")

# Разделим на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Оцениваем точность
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy * 100:.2f}%")

# Сохраняем модель
joblib.dump(model, MODEL_PATH)
print(f"Модель сохранена как {MODEL_PATH}")
