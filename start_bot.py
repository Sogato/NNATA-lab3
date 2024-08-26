import os
import joblib
import telebot
import numpy as np
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()
MODEL_PATH = 'neural_network/models/gradient_boosting_model.pkl'
MONTH_MAPPING = {
    'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5,
    'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11
}
DAY_MAPPING = {
    'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4, 'sat': 5, 'sun': 6
}

# Токен бота из переменной окружения
API_TOKEN = os.getenv('TELEGRAM_API_TOKEN')
if not API_TOKEN:
    raise ValueError("Токен API не найден. Убедитесь, что TELEGRAM_API_TOKEN установлен в .env файле.")

bot = telebot.TeleBot(API_TOKEN)
model = joblib.load(MODEL_PATH)


def preprocess_input(user_input):
    """
    Обрабатывает ввод пользователя и преобразует его в формат, пригодный для модели.

    :param user_input: строка, содержащая 12 значений, разделенных запятыми.
    :return: numpy array с преобразованными данными.
    """
    # Парсинг входных данных
    inputs = user_input.split(',')
    if len(inputs) != 12:
        raise ValueError("Неверное количество данных. Ожидается 12 значений.")

    # Преобразование числовых данных
    x_values = np.array(inputs[:10], dtype=float)

    # Обработка категориальных признаков с использованием Ordinal Encoding
    month = inputs[10].strip().lower()
    day = inputs[11].strip().lower()

    if month not in MONTH_MAPPING:
        raise ValueError(
            "Некорректное значение месяца. Ожидаются: jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec")
    if day not in DAY_MAPPING:
        raise ValueError("Некорректное значение дня недели. Ожидаются: mon, tue, wed, thu, fri, sat, sun")

    month_encoded = MONTH_MAPPING[month]
    day_encoded = DAY_MAPPING[day]
    final_input = np.concatenate([x_values, [month_encoded, day_encoded]])

    return final_input


def make_prediction(model, final_input):
    """
    Выполняет предсказание на основе модели и входных данных.

    :param model: обученная модель.
    :param final_input: numpy array с данными для предсказания.
    :return: предсказанная площадь пожара (в га).
    """
    prediction = model.predict([final_input])[0]
    prediction_area = np.expm1(prediction)  # Обратное преобразование log1p
    return prediction_area


def handle_start(message):
    """
    Обрабатывает команду /start.
    """
    bot.reply_to(message, "Добро пожаловать! Я бот для предсказания площади пожара. Введите данные в формате:\n"
                          "X, Y, FFMC, DMC, DC, ISI, temp, RH, wind, rain, month, day\n"
                          "Пример: 7, 5, 90.6, 35.4, 669.1, 6.7, 18.0, 33, 0.9, 0.0, oct, tue")


def handle_input(message):
    """
    Обрабатывает ввод пользователя и возвращает предсказание.
    """
    try:
        final_input = preprocess_input(message.text)
        prediction_area = make_prediction(model, final_input)
        bot.reply_to(message, f"Предсказанная площадь пожара: {prediction_area:.2f} га")

    except ValueError as e:
        bot.reply_to(message, f"Ошибка ввода: {e}")
    except Exception as e:
        bot.reply_to(message, f"Произошла ошибка: {e}")


# Привязка функций к событиям
@bot.message_handler(commands=['start'])
def send_welcome(message):
    handle_start(message)


@bot.message_handler(func=lambda message: True)
def process_message(message):
    handle_input(message)


bot.polling()
