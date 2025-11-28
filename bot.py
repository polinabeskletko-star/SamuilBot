import os
import asyncio
import random
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
from openai import OpenAI

# ==== SETTINGS ====

TOKEN = os.environ.get("BOT_TOKEN")
GROUP_CHAT_ID = os.environ.get("GROUP_CHAT_ID")
TIMEZONE = os.environ.get("BOT_TZ", "Australia/Brisbane")

# Optional: specific user in the group for jokes
TARGET_USER_ID_ENV = os.environ.get("TARGET_USER_ID")
TARGET_USER_ID = int(TARGET_USER_ID_ENV) if TARGET_USER_ID_ENV else None

# OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ---------- HELPERS ----------

def get_tz() -> pytz.BaseTzInfo:
    return pytz.timezone(TIMEZONE)


def compute_next_quarter(dt: datetime) -> datetime:
    """
    Next HH:15 after dt.
    Example: 09:02 -> 09:15, 09:20 -> 10:15, etc.
    dt must be timezone-aware.
    """
    next_run = dt.replace(minute=15, second=0, microsecond=0)
    if dt >= next_run:
        next_run = next_run + timedelta(hours=1)
    return next_run


def is_night_time(dt: datetime) -> bool:
    """
    Night time: 22:00–09:00 (inclusive of 22:00, exclusive of 09:00).
    Used ONLY for scheduled hourly question.
    """
    hour = dt.hour
    return hour >= 22 or hour < 9


# ---------- OPENAI JOKES ----------

FALLBACK_JOKES = [
    "Максим, я даже не знаю, что сказать… Ты сам понял, что написал? 😏",
    "Максим, опять текст уровня «гугл потом разберёт»? 😂",
    "Максим, это было задумано так или просто палец промахнулся по клавиатуре? 😉",
    "Читаю и думаю: это шедевр или черновик, который случайно отправился? 😄",
    "Максим, я сохранил это сообщение в папку «странные, но гениальные идеи». Ну или просто странные. 🤔",
]


async def call_openai_sarcastic_joke(user_text: str) -> str:
    """
    Call OpenAI to produce a short sarcastic reply in Russian.
    If OpenAI недоступен – берём случайную шутку из FALLBACK_JOKES.
    """
    # Если ключа нет – сразу рандомная шутка
    if not client or not OPENAI_API_KEY:
        print("OpenAI client is not configured, using local fallback joke.")
        return random.choice(FALLBACK_JOKES)

    system_prompt = (
        "Ты язвительный, но добрый друг Максима. "
        "Отвечаешь коротко на русском (до 25 слов), с лёгким стёбом, "
        "но не грубо и не обидно. Можно 1–2 смайлика, не больше."
    )

    user_prompt = (
        "В чате написали такое сообщение. Придумай одну короткую саркастичную реплику.\n\n"
        f"Сообщение: \"{user_text}\""
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=80,
            temperature=0.9,
        )

        reply = response.choices[0].message.content.strip()
        if not reply:
            raise ValueError("Empty reply from OpenAI")

        return reply

    except Exception as e:
        print("Error calling OpenAI, using fallback joke:", e)
        return random.choice(FALLBACK_JOKES)


# ---------- DEBUG LOGGER ----------

async def debug_logger(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Logs every update that reaches the bot.
    """
    chat = update.effective_chat
    user = update.effective_user
    msg = update.effective_message

    print(
        "DEBUG UPDATE:",
        f"chat_id={chat.id if chat else None}",
        f"chat_type={chat.type if chat else None}",
        f"user_id={user.id if user else None}",
        f"user_name={user.username if user else None}",
        f"text={repr(msg.text) if msg and msg.text else None}",
        flush=True,
    )


# ---------- COMMAND HANDLERS ----------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_type = update.effective_chat.type
    if chat_type == "private":
        await update.message.reply_text(
            "Привет! Я Друг Максима 🤖\n"
            "В группе я каждый час в 15 минут буду спрашивать:\n"
            "«Максим, как у тебя дела? Чем занимаешься?»\n"
            "Ночью с 22:00 до 9:00 я молчу 😴"
        )
    else:
        await update.message.reply_text(
            "Я отправляю вопрос Максиму каждый час в 15 минут, "
            "кроме ночи с 22:00 до 9:00."
        )


async def chat_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    await update.message.reply_text(
        f"Chat ID for this chat: `{cid}`",
        parse_mode="Markdown",
    )


async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user:
        await update.message.reply_text("Не могу определить тебя 🤷‍♂️")
        return

    text = (
        f"Твой user ID: `{user.id}`\n"
        f"Имя: {user.first_name or ''} {user.last_name or ''}"
    )
    if user.username:
        text += f"\nUsername: @{user.username}"

    await update.message.reply_text(text, parse_mode="Markdown")


# ---------- ECHO FOR PRIVATE CHATS ----------

async def echo_private(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type != "private":
        return
    text = update.message.text or ""
    await update.message.reply_text(f"Ты написал: {text}")


# ---------- TARGET USER LISTENER (SARCASTIC JOKES) ----------

async def target_user_listener(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    user = update.effective_user
    msg = update.effective_message

    # Only in group / supergroup
    if not chat or chat.type not in ("group", "supergroup"):
        return

    if not msg or not msg.text:
        return

    if TARGET_USER_ID is None:
        return

    if not user or user.id != TARGET_USER_ID:
        return

    user_text = msg.text.strip()
    if not user_text:
        return

    print(
        f"TARGET MESSAGE: from user {user.id} in chat {chat.id}: {repr(user_text)}",
        flush=True,
    )

    reply_text = await call_openai_sarcastic_joke(user_text)

    try:
        await msg.reply_text(reply_text)
        print("Sarcastic reply sent.", flush=True)
    except Exception as e:
        print("Error sending sarcastic reply:", e)


# ---------- SCHEDULED HOURLY MESSAGE ----------

async def hourly_message(context: ContextTypes.DEFAULT_TYPE):
    chat_id = GROUP_CHAT_ID
    if not chat_id:
        print("GROUP_CHAT_ID is not set; skipping hourly message.")
        return

    tz = get_tz()
    now = datetime.now(tz)

    if is_night_time(now):
        print(f"{now} – night time, message not sent.")
        return

    try:
        chat_id_int = int(chat_id)
        await context.bot.send_message(
            chat_id=chat_id_int,
            text="Максим, как у тебя дела? Чем занимаешься?"
        )
        print(f"{now} – hourly message sent to chat {chat_id_int}", flush=True)
    except Exception as e:
        print("Error sending hourly message:", e)


# ---------- MAIN APP ----------

def main():
    if not TOKEN:
        raise RuntimeError("BOT_TOKEN is not set in environment variables!")

    app = Application.builder().token(TOKEN).build()

    # DEBUG LOGGER – logs every update
    app.add_handler(MessageHandler(filters.ALL, debug_logger), group=0)

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("chatid", chat_id))
    app.add_handler(CommandHandler("whoami", whoami))

    # Private echo
    app.add_handler(
        MessageHandler(
            filters.TEXT & filters.ChatType.PRIVATE & ~filters.COMMAND,
            echo_private,
        ),
        group=1,
    )

    # Sarcastic jokes for target user in group
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            target_user_listener,
        ),
        group=2,
    )

    # JobQueue scheduling: every hour at HH:15
    job_queue = app.job_queue
    tz = get_tz()
    now = datetime.now(tz)
    first_run = compute_next_quarter(now)

    print(
        f"Local time now: {now} [{TIMEZONE}]. "
        f"First hourly_message scheduled at: {first_run} "
        f"(HH:15 each hour, skipping 22:00–09:00).",
        flush=True,
    )

    job_queue.run_repeating(
        hourly_message,
        interval=3600,
        first=first_run,
    )

    print("Bot started and hourly job scheduled...", flush=True)
    app.run_polling(allowed_updates=None)


if __name__ == "__main__":
    main()
