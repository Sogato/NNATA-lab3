import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

from neural_network.graphs import plot_feature_importances

FILE_PATH = "data/forestfires.csv"


def process_data(file_path):
    """
    Выполняет полный цикл обработки данных:
    - загрузка данных,
    - исследование данных,
    - предварительная обработка данных.

    :param file_path: Путь к файлу с данными.
    :return: Кортеж из четырех элементов: X_train, X_test, y_train, y_test.
    """
    data = pd.read_csv(file_path)

    print("Первые 5 строк датасета:")
    print(data.head())

    print("\nИнформация о датасете:")
    print(data.info())

    print("\nСтатистика по числовым признакам:")
    print(data.describe())

    # Категориальные признаки для кодирования
    categorical_columns = ['month', 'day']

    # Применение Ordinal Encoding для 'month' и 'day'
    encoder = OrdinalEncoder()
    data[categorical_columns] = encoder.fit_transform(data[categorical_columns])

    # Логарифмическое преобразование целевой переменной (с добавлением 1)
    data['log_area'] = np.log1p(data['area'])

    # Выделим признаки и целевую переменную
    X = data.drop(['area', 'log_area'], axis=1)
    y = data['log_area']

    # Масштабируем числовые признаки
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Получаем имена признаков после всех преобразований
    feature_names = X.columns

    print("\nРазмеры обучающей и тестовой выборок:")
    print(f"Обучающая выборка: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Тестовая выборка: X_test: {X_test.shape}, y_test: {y_test.shape}\n")

    return X_train, X_test, y_train, y_test, feature_names


def linear_regression(X_train, y_train, X_test, y_test):
    """
    Обучение и оценка линейной регрессии с исследованием гиперпараметров.
    """
    # Параметры для исследования
    param_grid = {
        'model': [LinearRegression(), Lasso(), Ridge()],
        'model__fit_intercept': [True, False],
    }

    # Создание пайплайна
    pipeline = Pipeline([
        ('model', LinearRegression())  # Заглушка, заменяется в GridSearchCV
    ])

    # GridSearchCV для подбора гиперпараметров
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Лучшая модель
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\nЛинейная регрессия (и её вариации):")
    print(f"Лучшие параметры: {grid_search.best_params_}")

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Среднеквадратичная ошибка: {mse}")
    print(f"Средняя абсолютная ошибка: {mae}")
    print(f"R^2: {r2}")

    # Сохранение модели
    joblib.dump(best_model, 'models/linear_model.pkl')
    print("Линейная модель сохранена в 'linear_model.pkl'")

    return best_model

def polynomial_regression(X_train, y_train, X_test, y_test):
    """
    Обучение и оценка полиномиальной регрессии с исследованием гиперпараметров.
    """
    # Параметры для GridSearchCV
    param_grid = {
        'poly_features__degree': [2, 3, 4, 5],  # Степени полинома
        'model': [LinearRegression(), Lasso(), Ridge()],
        'model__fit_intercept': [True, False],
    }

    # Создание пайплайна
    pipeline = Pipeline([
        ('poly_features', PolynomialFeatures()),  # Преобразование признаков в полиномиальные
        ('model', LinearRegression())  # Линейная регрессия (заглушка)
    ])

    # GridSearchCV для подбора гиперпараметров
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Лучшая модель
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\nПолиномиальная регрессия (и её вариации):")
    print(f"Лучшие параметры: {grid_search.best_params_}")

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Среднеквадратичная ошибка: {mse}")
    print(f"Средняя абсолютная ошибка: {mae}")
    print(f"R^2: {r2}")

    # Сохранение модели
    joblib.dump(best_model, 'models/polynomial_model.pkl')
    print("Полиномиальная модель сохранена в 'polynomial_model.pkl'")

    return best_model


def random_forest_regression(X_train, y_train, X_test, y_test, feature_names):
    """
    Обучение и оценка случайного леса с исследованием гиперпараметров.
    Сохранение модели в файл и визуализация важности признаков.
    """
    # Параметры для GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    rf = RandomForestRegressor(random_state=42)

    # GridSearchCV для подбора гиперпараметров
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Лучшая модель
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\nСлучайный лес:")
    print(f"Лучшие параметры: {grid_search.best_params_}")

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Среднеквадратичная ошибка: {mse}")
    print(f"Средняя абсолютная ошибка: {mae}")
    print(f"R^2: {r2}")

    # Визуализация важности признаков
    importances = best_model.feature_importances_
    plot_feature_importances(importances, feature_names,
                             "Значимые признаки - Случайный лес",
                             "../img/graphs_img/random_forest_importances.png")
    print("График важности признаков для случайного леса сохранен в 'random_forest_importances.png'")

    # Сохранение модели
    joblib.dump(best_model, 'models/random_forest_model.pkl')
    print("Модель случайного леса сохранена в 'random_forest_model.pkl'")

    return best_model

def gradient_boosting_regression(X_train, y_train, X_test, y_test, feature_names):
    """
    Обучение и оценка Gradient Boosting Regressor с исследованием гиперпараметров.
    Сохранение модели в файл и визуализация важности признаков.
    """
    # Параметры для GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0]
    }

    gb = GradientBoostingRegressor(random_state=42)

    # GridSearchCV для подбора гиперпараметров
    grid_search = GridSearchCV(estimator=gb, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Лучшая модель
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\nGradient Boosting Regressor:")
    print(f"Лучшие параметры: {grid_search.best_params_}")

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Среднеквадратичная ошибка: {mse}")
    print(f"Средняя абсолютная ошибка: {mae}")
    print(f"R^2: {r2}")

    # Визуализация важности признаков
    importances = best_model.feature_importances_
    plot_feature_importances(importances, feature_names,
                             "Значимые признаки - Gradient Boosting",
                             "../img/graphs_img/gradient_boosting_importances.png")
    print("График важности признаков для Gradient Boosting сохранен в 'gradient_boosting_importances.png'")

    # Сохранение модели
    joblib.dump(best_model, 'models/gradient_boosting_model.pkl')
    print("Модель Gradient Boosting сохранена в 'gradient_boosting_model.pkl'")

    return best_model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names = process_data(FILE_PATH)

    # Линейная регрессия
    linear_model = linear_regression(X_train, y_train, X_test, y_test)
    # Полиномиальная регрессия
    polynomial_model = polynomial_regression(X_train, y_train, X_test, y_test)
    # Случайный лес
    random_forest_model = random_forest_regression(X_train, y_train, X_test, y_test, feature_names)
    # Gradient Boosting Regressor
    gradient_boosting_model = gradient_boosting_regression(X_train, y_train, X_test, y_test, feature_names)
