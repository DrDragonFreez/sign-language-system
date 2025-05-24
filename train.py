import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# Загрузка данных с low_memory=False для предотвращения предупреждений
df = pd.read_csv("dataset.csv", header=None, low_memory=False)

# Проверка и обработка пропусков (NaN)
print("Пропуски по столбцам:\n", df.isnull().sum())
df = df.dropna()  # можно заменить на df.fillna(df.mean()), если хочешь заполнить

X = df.iloc[:, :-1].values  # 126 признаков
y = df.iloc[:, -1].values   # Метка (название жеста)

print(f" Примеры: {X.shape[0]}, признаков на пример: {X.shape[1]} (должно быть 126)")

# Деление на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Обучение модели
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Проверка точности
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f" Точность модели: {accuracy * 100:.2f}%")

# Сохранение модели
joblib.dump(model, "model.pkl")
print(" Модель сохранена как model.pkl")
