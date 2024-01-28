# Применяя различные методы машинного обучения, оценить их производительность и провести анализ результатов
# Загрузить набор данных о населении городов и создайте модель регрессии для прогнозирования роста населения в будущем


# Планируется использовать CatBoost, поэтому импортируется из библиотеки catboost модуль CatBoostRegressor
from catboost import CatBoostRegressor

# Планируется использовать XGBoost, поэтому импортируются из библиотеки lightgbm модуль LGBMRegressor
from xgboost import XGBRegressor

# Планировался использоваться LightGBM, но совместно с sci-learn модуль lightgbm не устанавливается из-за конфликтов
# from lightgbm import LGBMRegressor

# Планируется также использовать модели Linear Regression, Lasso Regression, Ridge Regression, ElasticNet Regression,
# Random Forest Regressor, Gradient Boosting Regressor, SVR, AdaBoostRegressor, KNeighborsRegressor, поэтому импортируются
# из библиотеки sklearn следующие модули:
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import pandas as pd
from prepare_data_2 import df_zaragoza, df_malaga, df_murcia, df_las_palmas
import matplotlib.pyplot as plt

X = df_zaragoza.iloc[1:, 1:]
y = df_zaragoza.iloc[1:, 0]

# Визуализация исходных
plt.figure(figsize=(8, 6))
plt.scatter(y.index, y, c="b")
plt.xlabel("годы")
plt.ylabel("Население")
plt.title(f"Исходные данные {df_zaragoza.iloc[0, 0]}")
plt.xticks(rotation=90)
plt.show()
# На графике видно, что за 1989 и 1997 годы нет данных, поэтому их необходимо удалить
# Удаление строк с пропущенными значениями и заново определение X и y
df_zaragoza = df_zaragoza.drop(["1989", "1997"], axis=0)
X = df_zaragoza.iloc[1:, 1:]
y = df_zaragoza.iloc[1:, 0]

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Нормализация признаков
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Обучение модели регрессии (SVR)
# model = SVR(kernel='linear', C=1)

# Обучение модели регрессии (Linear Regression)
# model = LinearRegression()

# Обучение модели регрессии (Lasso Regression)
# model = Lasso(alpha=0.1)

# Обучение модели регрессии (Ridge Regression)
# model = Ridge(alpha=1.0)

# Обучение модели регрессии (ElasticNet Regression)
# model = ElasticNet(alpha=0.1, l1_ratio=0.5)

# Обучение модели регрессии (Random Forest Regressor)
# model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

# Обучение модели регрессии (Gradient Boosting Regressor)
# model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)

# Обучение модели регрессии (XGBoost)
# model = XGBRegressor(n_estimators=100, max_depth=5, random_state=42)

# Обучение модели регрессии (CatBoost)
# model = CatBoostRegressor(n_estimators=100, max_depth=5, random_state=42)

# Обучение модели регрессии (LightGBM)
# model = LGBMRegressor(n_estimators=100, max_depth=5, random_state=42)

# Обучение модели регрессии (AdaBoost)
# model = AdaBoostRegressor(n_estimators=100, random_state=42)

# Обучение модели регрессии (KNeighborsRegressor)
model = KNeighborsRegressor(n_neighbors=5)


model.fit(X_train, y_train)
# Предсказание на тестовом наборе
y_pred = model.predict(X_test)
# Метрики качества
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)
# Оцениваем качество модели с помощью средней абсолютной ошибки (MAE) для модуля KNeighborsRegressor
# mae = mean_absolute_error(y_test, y_pred)
# print(f"Средняя абсолютная ошибка (MAE): {mae:.2f}")

# Визуализация результатов
plt.figure(figsize=(8, 6))
plt.scatter(y_test.index, y_pred, c="r")
plt.scatter(y_test.index, y_test, c="b")
plt.xlabel("годы")
plt.ylabel("Население")
plt.title(f"Сравнение точности модели для данных {df_zaragoza.iloc[0, 0]}")
plt.legend(["Предсказанные значения", "Реальные значения"])
plt.show()
