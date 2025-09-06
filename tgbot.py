import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import logging
import os
import json
from dotenv import load_dotenv
from io import StringIO, BytesIO
from telegram import Update, InputFile, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Константы для состояний ConversationHandler
INPUT_FEATURES, INPUT_VALUES = range(2)

# Загрузка обученной модели
MODEL_PATH = r''
try:
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {e}")
    raise

# Топ-10 важных признаков
TOP_FEATURES = [
    'savings_sum_sa_now',
    'savings_avg_sa_1m',
    'savings_sum_sa_1m',
    'savings_sum_sa_debet_1m',
    'avg_dep_avg_balance_fact_1month_amt_term_savings',
    'sum_dep_expense_3month_amt_term_savings',
    'sum_dep_income_3month_amt_term_savings',
    'sum_dep_expense_1month_amt_term_savings',
    'sum_dep_income_1month_amt_term_savings',
    'savings_sum_sa_2m'
]

# Клавиатура для главного меню
main_menu_keyboard = [
    ['Ввести данные вручную', 'Загрузить CSV файл'],
    ['Помощь', 'Отмена']
]

# Клавиатура для отмены
cancel_keyboard = [['Отмена']]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет приветственное сообщение и инструкции."""
    welcome_text = """
    Привет! Я бот для предсказания 50-го перцентиля распределения суммарных остатков на накопительных счетах.

    Вы можете:
    1. Ввести значения для топ-10 важных признаков вручную
    2. Загрузить CSV файл со всеми признаками (в нём должны быть минимум 10 важных признаков)
    """
    reply_markup = ReplyKeyboardMarkup(main_menu_keyboard, resize_keyboard=True)
    await update.message.reply_text(welcome_text, reply_markup=reply_markup)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет справку по командам."""
    help_text = """
    Доступные команды:
    /start - начать работу с ботом
    /help - показать это сообщение
    /input_features - ввести значения признаков вручную
    /cancel - отменить текущую операцию

    Или используйте кнопки:
    - Ввести данные вручную: начать ручной ввод признаков
    - Загрузить CSV файл: отправить файл с данными
    - Помощь: показать это сообщение
    - Отмена: отменить текущую операцию
    """
    reply_markup = ReplyKeyboardMarkup(main_menu_keyboard, resize_keyboard=True)
    await update.message.reply_text(help_text, reply_markup=reply_markup)


async def input_features(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начинает процесс ручного ввода признаков."""
    features_text = "Пожалуйста, введите значения для следующих признаков (по одному в строке):\n\n"
    features_text += "\n".join([f"{i + 1}. {feature}" for i, feature in enumerate(TOP_FEATURES)])

    reply_markup = ReplyKeyboardMarkup(cancel_keyboard, resize_keyboard=True)
    await update.message.reply_text(features_text, reply_markup=reply_markup)
    return INPUT_VALUES


async def process_feature_values(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обрабатывает введенные значения признаков."""
    user_input = update.message.text

    # Обработка команды отмены через кнопку
    if user_input.lower() == 'отмена':
        return await cancel(update, context)

    values = []

    # Обрабатываем каждую строку ввода
    for line in user_input.split('\n'):
        if line.strip():
            # Пробуем разделить по двоеточию
            parts = line.split(':', 1)  # Разделяем только по первому двоеточию
            value = parts[-1].strip() if len(parts) > 1 else line.strip()
            values.append(value if value else np.nan)

    if len(values) != len(TOP_FEATURES):
        await update.message.reply_text(f"Ожидается {len(TOP_FEATURES)} значений.")
        return INPUT_VALUES

    try:
        # Создаем DataFrame и обрабатываем данные
        input_data = pd.DataFrame([values], columns=TOP_FEATURES)

        for col in TOP_FEATURES:
            # Пропускаем уже NaN значения
            if pd.isna(input_data[col].iloc[0]):
                input_data[col] = numerical_medians.get(col, 0)
                continue

            # Пытаемся преобразовать в число
            try:
                input_data[col] = pd.to_numeric(input_data[col], errors='raise')
            except ValueError:
                # Если не число - используем медиану
                input_data[col] = numerical_medians.get(col, 0)
                logger.info(f"Некорректное значение для {col}, использована медиана")

        prediction = make_prediction(input_data, numerical_medians)
        reply_markup = ReplyKeyboardMarkup(main_menu_keyboard, resize_keyboard=True)
        await update.message.reply_text(
            f"Предсказанный 50-й перцентиль: {prediction:.2f} рублей",
            reply_markup=reply_markup
        )

    except Exception as e:
        await update.message.reply_text(f"Ошибка: {str(e)}")
        return INPUT_VALUES

    return ConversationHandler.END


async def process_csv(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает загруженный CSV файл."""
    try:
        # Получаем файл
        file = await context.bot.get_file(update.message.document.file_id)
        file_bytes = await file.download_as_bytearray()

        # Читаем CSV
        csv_data = StringIO(file_bytes.decode('utf-8'))
        input_data = pd.read_csv(csv_data)

        # Проверяем наличие необходимых столбцов
        missing_features = set(TOP_FEATURES) - set(input_data.columns)
        if missing_features:
            await update.message.reply_text(
                f"Ошибка: в файле отсутствуют следующие важные признаки: {', '.join(missing_features)}"
            )
            return

        # Делаем предсказания для всех строк
        predictions = []
        for _, row in input_data.iterrows():
            prediction = make_prediction(row.to_frame().T, numerical_medians)
            predictions.append(prediction)

        # Добавляем предсказания в DataFrame
        input_data['predicted_50_percentile'] = predictions

        # Создаем CSV с результатами
        output = StringIO()
        input_data.to_csv(output, index=False)
        output.seek(0)

        # Отправляем файл обратно пользователю
        reply_markup = ReplyKeyboardMarkup(main_menu_keyboard, resize_keyboard=True)
        await update.message.reply_document(
            document=InputFile(BytesIO(output.getvalue().encode('utf-8')), filename='predictions.csv'),
            caption="Результаты предсказаний",
            reply_markup=reply_markup
        )

    except Exception as e:
        logger.error(f"Ошибка обработки CSV: {e}")
        await update.message.reply_text(f"Ошибка обработки файла: {str(e)}")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает текстовые сообщения с кнопок."""
    text = update.message.text
    if text == 'Ввести данные вручную':
        await input_features(update, context)
    elif text == 'Загрузить CSV файл':
        await update.message.reply_text("Пожалуйста, загрузите CSV файл с данными.")
    elif text == 'Помощь':
        await help_command(update, context)
    elif text == 'Отмена':
        await cancel(update, context)


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Отменяет текущую операцию."""
    reply_markup = ReplyKeyboardMarkup(main_menu_keyboard, resize_keyboard=True)
    await update.message.reply_text(
        'Операция отменена.',
        reply_markup=reply_markup
    )
    return ConversationHandler.END


# Загрузка статистик из файла
with open(r'') as f:
    train_stats = json.load(f)

# Получаем медианы для числовых признаков
numerical_medians = train_stats["numerical"]


def make_prediction(input_data, numerical_medians):
    """Делает предсказание, заменяя пропуски (NaN/None) на медианные значения"""
    # 1. Создаем DataFrame со всеми признаками модели
    full_features = model.feature_names_
    input_df = pd.DataFrame(columns=full_features)

    # 2. Заполняем переданные пользователем признаки
    for feature in TOP_FEATURES:
        if feature in input_data:
            # Получаем значение из input_data
            value = input_data[feature]

            # Проверяем на NaN/None (даже если передан Series/DataFrame)
            if pd.isna(value).any():
                input_df[feature] = [numerical_medians.get(feature, 0)]
            else:
                input_df[feature] = [value]
        else:
            # Если признак отсутствует в input_data
            input_df[feature] = [numerical_medians.get(feature, 0)]

    # 3. Заполняем остальные признаки медианами
    for feature in full_features:
        if feature not in TOP_FEATURES:
            input_df[feature] = [numerical_medians.get(feature, 0)]

    # 4. Делаем предсказание
    prediction = np.expm1(model.predict(input_df))[0]
    return prediction


def main() -> None:
    """Запуск бота."""
    load_dotenv()  # Загружает переменные из .env

    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        raise ValueError("Токен бота не найден в .env файле!")

    application = Application.builder().token(token).build()

    # Обработчики команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("cancel", cancel))

    # ConversationHandler для ручного ввода признаков
    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler('input_features', input_features),
            MessageHandler(filters.Regex('^Ввести данные вручную$'), input_features)
        ],
        states={
            INPUT_VALUES: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_feature_values),
            ],
        },
        fallbacks=[
            CommandHandler('cancel', cancel),
            MessageHandler(filters.Regex('^Отмена$'), cancel)
        ],
    )
    application.add_handler(conv_handler)

    # Обработчик CSV файлов
    application.add_handler(MessageHandler(filters.Document.FileExtension("csv"), process_csv))

    # Обработчик текстовых сообщений (для кнопок)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Запускаем бота
    application.run_polling()


if __name__ == '__main__':
    main()