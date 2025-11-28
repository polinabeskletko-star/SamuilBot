import os
from datetime import datetime, timedelta

import pytz
from openai import OpenAI
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ================== SETTINGS / ENV VARS ==================

# Telegram bot token
TOKEN = os.environ.get("BOT_TOKEN")

# Group chat ID where hourly question will be sent (e.g. "-1001234567890")
GROUP_CHAT_ID = os.environ.get("GROUP_CHAT_ID")

# Timezone (default: Brisbane)
TIMEZONE = os.environ.get("BOT_TZ", "Australia/Brisbane")

# Target user and chat for sarcastic replies
TARGET_USER_ID_ENV = os.environ.get("TARGET_USER_ID")   # numeric string
TARGET_CHAT_ID = os.environ.get("TARGET_CHAT_ID")       # string chat id

TARGET_USER_ID = int(TARGET_USER_ID_ENV) if TARGET_USER_ID_ENV else None

# Second user: поддержка и усиление (по умолчанию 502791142)
SUPPORT_USER_ID_ENV = os.environ.get("SUPPORT_USER_ID")
SUPPORT_USER_ID = (
    int(SUPPORT_USER_ID_ENV)
    if SUPPORT_USER_ID_ENV
    else 502791142  # твой запрошенный ID по умолчанию
)

# OpenAI
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI()  # API key берётся из OPENAI_API_KEY


# ================== HELPERS ==================

def get_tz() -> pytz.BaseTzInfo:
    """Return timezone object from TIMEZONE setting."""
    return pytz.timezone(TIMEZONE)


def compute_next_quarter_hour(dt: datetime) -> datetime:
    """
    Return the next time at HH:15 after the given datetime `dt`.
    `dt` must be timezone-aware.
    Example: 09:02 -> 09:15, 09:20 -> 10:15, etc.
    """
    next_run = dt.replace(minute=15, second=0, microsecond=0)
    if dt >= next_run:
        next_run = next_run + timedelta(hours=1)
    return next_run


def is_night_time(dt: datetime) -> bool:
    """
    Night time = 22:00–09:00 (inclusive 22:00, exclusive 09:00).
    During this time the bot will NOT send the hourly question.
    """
    hour = dt.hour
    return hour >= 22 or hour < 9


def describe_part_of_day_ru(dt: datetime) -> str:
    """Return Russian description of time of day."""
    hour = dt.hour
    if 9 <= hour < 12:
        return "утро"
    elif 12 <= hour < 18:
        return "день"
    elif 18 <= hour < 22:
        return "вечер"
    else:
        return "ночь"


def build_hourly_prompt(now: datetime) -> str:
    """Prompt для генерации ежечасного вопроса к Максиму."""
    weekday_names = [
        "понедельник",
        "вторник",
        "среда",
        "четверг",
        "пятница",
        "суббота",
        "воскресенье",
    ]
    weekday = weekday_names[now.weekday()]
    part_of_day = describe_part_of_day_ru(now)

    return (
        "Сгенерируй ОДИН короткий вопрос по-русски для телеграм-чата, "
        "обращаясь к Максиму по имени. "
        "Смысл: узнать, как у него дела и чем он сейчас занимается. "
        "Стиль: дружелюбный, чуть-чуть шутливый, но без грубостей. "
        "Не пиши смайлики и не используй хэштеги. "
        "Упомяни в формулировке, что сейчас " + part_of_day +
        " и " + weekday + ". "
        "Максимум 20 слов. Только текст вопроса, без пояснений."
    )


def build_sarcastic_prompt(user_text: str) -> str:
    """Prompt для саркастического ответа на сообщение Максима."""
    return (
        "Ты язвительный, но доброжелательный друг в телеграм-чате. "
        "Ответь на сообщение короткой шутливой фразой по-русски. "
        "Стиль: лёгкий сарказм, без оскорблений, без мата, максимум 25 слов. "
        "Не используй смайлики и хэштеги. "
        "Сообщение пользователя:\n\n"
        f"{user_text}\n\n"
        "Теперь придумай один подходящий саркастический ответ. Только ответ, без пояснений."
    )


def build_supportive_prompt(user_text: str) -> str:
    """Prompt для поддерживающего/усиливающего ответа на сообщение второго пользователя."""
    return (
        "Ты очень поддерживающий и воодушевляющий друг в телеграм-чате. "
        "Ответь на сообщение короткой фразой по-русски, которая поддерживает, "
        "усиливает и хвалит собеседника. "
        "Стиль: тёплый, мотивирующий, без пафоса, максимум 25 слов. "
        "Не используй смайлики и хэштеги. "
        "Сообщение пользователя:\n\n"
        f"{user_text}\n\n"
        "Теперь придумай один подходящий поддерживающий ответ. Только ответ, без пояснений."
    )


def generate_ai_text(prompt: str, fallback: str) -> str:
    """
    Вспомогательная функция: вызвать OpenAI Responses API и вернуть текст.
    В случае ошибки вернёт fallback и напечатает ошибку в логи.
    """
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
        )
        # Структура ответа Responses API: output[0].content[0].text
        if resp.output and resp.output[0].content:
            text = resp.output[0].content[0].text.strip()
            if text:
                return text
    except Exception as e:
        print("Error calling OpenAI, using fallback text:", e)

    return fallback


# ================== COMMAND HANDLERS ==================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    chat_type = update.effective_chat.type
    if chat_type == "private":
        await update.message.reply_text(
            "Привет! Я Друг Максима 🤖\n"
            "В группе я каждый час в :15 буду спрашивать, как у Максима дела,\n"
            "формулировки будут разными и зависят от времени суток.\n"
            "Ночью с 22:00 до 9:00 я молчу 😴\n"
            "А ещё я отвечаю Максиму с лёгким сарказмом и поддерживаю другого выбранного пользователя."
        )
    else:
        await update.message.reply_text(
            "Я отправляю вопрос Максиму каждый час в :15 с разными формулировками, "
            "кроме ночи с 22:00 до 9:00. "
            "Также шучу над одним пользователем и поддерживаю другого 😊"
        )


async def chat_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send back the current chat ID (useful to configure GROUP_CHAT_ID / TARGET_CHAT_ID)."""
    cid = update.effective_chat.id
    await update.message.reply_text(
        f"Chat ID for this chat: `{cid}`",
        parse_mode="Markdown"
    )


async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Return user id for testing TARGET_USER_ID / SUPPORT_USER_ID."""
    user = update.effective_user
    if not user:
        return
    await update.message.reply_text(f"Your user id: `{user.id}`", parse_mode="Markdown")


async def echo_private(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Simple echo reply ONLY in private chats.
    In groups the bot stays quiet (except scheduled messages + target jokes/support).
    """
    if update.effective_chat.type != "private":
        return

    text = update.message.text
    await update.message.reply_text(f"Ты написал: {text}")


# ================== GROUP MESSAGE HANDLER (JOKES & SUPPORT) ==================

async def handle_group_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обрабатываем сообщения в группах.
    Если сообщение от TARGET_USER_ID в TARGET_CHAT_ID – сарказм через OpenAI.
    Если сообщение от SUPPORT_USER_ID – поддерживающий ответ через OpenAI.
    Остальные пользователи игнорируются.
    """
    message = update.message
    if not message:
        return

    chat = update.effective_chat
    user = update.effective_user
    text = message.text or ""

    chat_id_str = str(chat.id)
    user_id = user.id if user else None
    user_name = user.username if user and user.username else (user.full_name if user else "Unknown")

    print(
        f"DEBUG UPDATE: chat_id={chat.id} chat_type={chat.type} "
        f"user_id={user_id} user_name={user_name} text='{text}'"
    )

    # Ограничиваемся целевым чатом (если задан)
    if TARGET_CHAT_ID and chat_id_str != TARGET_CHAT_ID:
        return

    if user_id is None:
        return

    # ----- Ветка 1: сарказм для TARGET_USER_ID -----
    if TARGET_USER_ID is not None and user_id == TARGET_USER_ID:
        print(
            f"TARGET (sarcastic) MESSAGE: from user {user_id} in chat {chat.id}: '{text}'"
        )

        prompt = build_sarcastic_prompt(text)
        fallback = "Интересно, это ты сейчас серьёзно или опять шутишь?"
        reply_text = generate_ai_text(prompt, fallback)

        try:
            await message.reply_text(reply_text)
            print("Sarcastic reply sent.")
        except Exception as e:
            print("Error sending sarcastic reply:", e)
        return

    # ----- Ветка 2: поддержка для SUPPORT_USER_ID -----
    if SUPPORT_USER_ID is not None and user_id == SUPPORT_USER_ID:
        print(
            f"SUPPORT (encouraging) MESSAGE: from user {user_id} in chat {chat.id}: '{text}'"
        )

        prompt = build_supportive_prompt(text)
        fallback = "Звучит очень круто, продолжай в том же духе, это реально впечатляет!"
        reply_text = generate_ai_text(prompt, fallback)

        try:
            await message.reply_text(reply_text)
            print("Supportive reply sent.")
        except Exception as e:
            print("Error sending supportive reply:", e)
        return

    # Остальные пользователи — игнор
    return


# ================== SCHEDULED HOURLY MESSAGE ==================

async def hourly_message(context: ContextTypes.DEFAULT_TYPE):
    """
    Ежечасное сообщение в GROUP_CHAT_ID в HH:15,
    но только если не ночь (22:00–09:00).
    Текст формируется через OpenAI, чтобы фразы отличались и учитывали время суток.
    """
    chat_id = GROUP_CHAT_ID
    if not chat_id:
        print("GROUP_CHAT_ID is not set; skipping hourly message.")
        return

    tz = get_tz()
    now = datetime.now(tz)

    if is_night_time(now):
        print(f"{now} – night time, hourly message not sent.")
        return

    # Prompt для OpenAI
    prompt = build_hourly_prompt(now)
    fallback = "Максим, как у тебя дела? Чем занимаешься сейчас?"

    text = generate_ai_text(prompt, fallback)

    try:
        chat_id_int = int(chat_id)
        await context.bot.send_message(
            chat_id=chat_id_int,
            text=text
        )
        print(f"{now} – hourly AI message sent to chat {chat_id_int}: {text}")
    except Exception as e:
        print("Error sending hourly message:", e)


# ================== MAIN APP ==================

def main():
    if not TOKEN:
        raise RuntimeError("BOT_TOKEN is not set in environment variables!")

    app = Application.builder().token(TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("chatid", chat_id))
    app.add_handler(CommandHandler("whoami", whoami))

    # Private echo
    app.add_handler(
        MessageHandler(
            filters.TEXT & filters.ChatType.PRIVATE & ~filters.COMMAND,
            echo_private,
        )
    )

    # Group messages (for sarcastic + supportive replies)
    app.add_handler(
        MessageHandler(
            filters.TEXT & filters.ChatType.GROUPS & ~filters.COMMAND,
            handle_group_message,
        )
    )

    # JobQueue scheduling (HH:15 every hour)
    job_queue = app.job_queue
    tz = get_tz()
    now = datetime.now(tz)
    first_run = compute_next_quarter_hour(now)

    print(
        f"Local time now: {now} [{TIMEZONE}]. "
        f"First hourly_message scheduled at: {first_run} "
        f"(HH:15 each hour, skipping 22:00–09:00)."
    )

    job_queue.run_repeating(
        hourly_message,
        interval=3600,   # every hour
        first=first_run,
    )

    print(
        "Bot started and hourly AI job scheduled...\n"
        f"TARGET_USER_ID (sarcasm): {TARGET_USER_ID}, "
        f"SUPPORT_USER_ID (support): {SUPPORT_USER_ID}, "
        f"TARGET_CHAT_ID: {TARGET_CHAT_ID}"
    )
    app.run_polling()


if __name__ == "__main__":
    main()