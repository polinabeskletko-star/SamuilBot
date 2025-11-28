import os
from datetime import datetime, timedelta

import pytz
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ==== НАСТРОЙКИ ====

TOKEN = os.environ.get("BOT_TOKEN")
GROUP_CHAT_ID = os.environ.get("GROUP_CHAT_ID")  # например "-1234567890"
# Часовой пояс – можно переопределить через переменную окружения BOT_TZ
TIMEZONE = os.environ.get("BOT_TZ", "Australia/Brisbane")


# ---------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ----------

def get_tz() -> pytz.BaseTzInfo:
    """Возвращает объект часового пояса."""
    return pytz.timezone(TIMEZONE)


def seconds_until_next_quarter() -> float:
    """
    Считает, через сколько секунд наступит ближайшее время HH:15
    в выбранном часовом поясе.
    """
    tz = get_tz()
    now = datetime.now(tz)
    # ближайшее время с минутой 15
    next_run = now.replace(minute=15, second=0, microsecond=0)
    if now >= next_run:
        # если уже позже 15-й минуты, переносим на следующий час
        next_run = next_run + timedelta(hours=1)
    delta = next_run - now
    return delta.total_seconds()


def is_night_time(dt: datetime) -> bool:
    """Ночной период с 22:00 до 9:00 включительно (по локальному времени)."""
    hour = dt.hour
    return hour >= 22 or hour < 9


# ---------- КОМАНДЫ ----------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_type = update.effective_chat.type
    if chat_type == "private":
        await update.message.reply_text(
            "Привет! Я Друг Максима 🤖\n"
            "В группе раз в час в 15 минут буду спрашивать:\n"
            "«Максим, как у тебя дела? Чем занимаешься?»\n"
            "Ночью с 22:00 до 9:00 я молчу 😴"
        )
    else:
        await update.message.reply_text(
            "Я настроен отправлять вопрос Максиму каждый час в 15 минут, "
            "кроме ночи с 22:00 до 9:00."
        )


async def chat_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    await update.message.reply_text(
        f"Chat ID для этого чата: `{cid}`",
        parse_mode="Markdown"
    )


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Эхо-ответ ТОЛЬКО в личке."""
    if update.effective_chat.type != "private":
        return
    text = update.message.text
    await update.message.reply_text(f"Ты написал: {text}")


# ---------- ПОЧАСОВОЕ СООБЩЕНИЕ ----------

async def hourly_message(context: ContextTypes.DEFAULT_TYPE):
    """Отправляет сообщение в группу в 15 минут каждого часа, кроме ночи."""
    chat_id = GROUP_CHAT_ID
    if not chat_id:
        print("GROUP_CHAT_ID не задан; пропускаю отправку.")
        return

    tz = get_tz()
    now = datetime.now(tz)

    if is_night_time(now):
        print(f"{now} – ночное время, сообщение не отправлено.")
        return

    try:
        chat_id_int = int(chat_id)
        await context.bot.send_message(
            chat_id=chat_id_int,
            text="Максим, как у тебя дела? Чем занимаешься?"
        )
        print(f"{now} – отправлено сообщение в чат {chat_id_int}")
    except Exception as e:
        print("Error sending hourly message:", e)


def main():
    if not TOKEN:
        raise RuntimeError("BOT_TOKEN is not set in environment variables!")

    app = Application.builder().token(TOKEN).build()

    # Команды
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("chatid", chat_id))

    # Эхо только в личных чатах
    app.add_handler(
        MessageHandler(
            filters.TEXT & filters.ChatType.PRIVATE & ~filters.COMMAND,
            echo,
        )
    )

    # Планировщик
    job_queue = app.job_queue
    first_delay = seconds_until_next_quarter()
    print(f"First run in {first_delay:.0f} seconds.")
    job_queue.run_repeating(
        hourly_message,
        interval=3600,       # раз в час
        first=first_delay,   # первый запуск в ближайшие HH:15
    )

    print("Bot started and hourly job scheduled...")
    app.run_polling()


if __name__ == "__main__":
    main()
