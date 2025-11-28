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

# ===========================
# ENVIRONMENT VARIABLES
# ===========================
TOKEN = os.environ.get("BOT_TOKEN")
GROUP_CHAT_ID = os.environ.get("GROUP_CHAT_ID")          # e.g. "-1001234567890"
TIMEZONE = os.environ.get("BOT_TZ", "Australia/Brisbane")
TARGET_USER_ID = os.environ.get("TARGET_USER_ID")        # e.g. "123456789"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# OpenAI client (optional – jokes work only if key is set)
client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)


# ===========================
# TIME HELPERS
# ===========================
def get_tz():
    return pytz.timezone(TIMEZONE)


def compute_next_quarter(dt: datetime) -> datetime:
    """
    Return next HH:15 time.
      09:05 → 09:15
      09:20 → 10:15
    """
    next_run = dt.replace(minute=15, second=0, microsecond=0)
    if dt >= next_run:
        next_run = next_run + timedelta(hours=1)
    return next_run


def is_night_time(dt: datetime) -> bool:
    """Night = 22:00–09:00 (no posts)."""
    return dt.hour >= 22 or dt.hour < 9


# ===========================
# OPENAI JOKE GENERATOR
# ===========================
async def generate_sarcastic_joke(user_text: str) -> str:
    """
    Generate a short sarcastic joke in Russian about the user's message.
    If OpenAI isn't configured, return a fallback text.
    """
    if client is None:
        return (
            "Я бы сейчас выдал гениальную саркастичную шутку, "
            "но меня ещё не подключили к мозгам (OPENAI_API_KEY) 🤖"
        )

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            instructions=(
                "Ты саркастичный, но добрый друг Максима. "
                "Отвечай на русском, коротко (1–2 предложения), язвительно, "
                "но без мата и откровенных оскорблений. "
                "Цель — подшутить, а не обидеть."
            ),
            input=(
                "Максим написал в чат:\n"
                f"{user_text}\n\n"
                "Ответь одной короткой саркастичной шуткой-комментарием."
            ),
        )

        # High-level helper that returns concatenated text output
        joke = response.output_text.strip()
        if len(joke) > 500:
            joke = joke[:500]
        return joke

    except Exception as e:
        print("Error calling OpenAI:", e)
        return "Сегодня сарказм на техническом обслуживании, приходи позже 😅"


# ===========================
# BOT COMMAND HANDLERS
# ===========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type == "private":
        await update.message.reply_text(
            "Привет! Я Друг Максима 🤖\n"
            "• В группе каждый час в 15 минут спрашиваю: "
            "«Максим, как у тебя дела? Чем занимаешься?»\n"
            "• Ночью с 22:00 до 09:00 молчу 😴\n"
            "• На сообщения одного выбранного человека в группе отвечаю "
            "саркастичными шутками, но шепотом — в личку 😉\n"
            "Команда /myid покажет твой user ID, /chatid — ID чата."
        )
    else:
        await update.message.reply_text(
            "Я буду писать каждый час в 15 минут (кроме 22:00–09:00), "
            "а шутки для Максима отправляю ему в личку."
        )


async def chat_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    await update.message.reply_text(f"Chat ID: `{cid}`", parse_mode="Markdown")


async def myid(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    await update.message.reply_text(f"Your user ID: `{uid}`", parse_mode="Markdown")


async def echo_private(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Echo only in private chats; no spam in groups."""
    if update.effective_chat.type != "private":
        return
    await update.message.reply_text(f"Ты написал: {update.message.text}")


# ===========================
# GROUP MESSAGE HANDLER (JOKES)
# ===========================
async def handle_group_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Called for every TEXT message in groups.
    Logic:
    - Only process messages from the configured GROUP_CHAT_ID (if set)
    - Only process if sender's id == TARGET_USER_ID
    - Generate sarcastic joke and send it to user's private chat
    """
    msg = update.message
    if not msg or not msg.text:
        return

    chat = update.effective_chat
    user = update.effective_user

    # Only in the configured group
    if GROUP_CHAT_ID and str(chat.id) != str(GROUP_CHAT_ID):
        return

    # Ignore commands like /start etc.
    if msg.text.startswith("/"):
        return

    # Only for the target user
    if TARGET_USER_ID and str(user.id) != str(TARGET_USER_ID):
        return

    print(f"Got message from target user {user.id} in group {chat.id}: {msg.text!r}")

    joke = await generate_sarcastic_joke(msg.text)

    # Reply in private (B)
    try:
        await context.bot.send_message(
            chat_id=user.id,
            text=joke,
        )
        print(f"Sent private joke to user {user.id}")
    except Exception as e:
        # Common issue: user never started bot in private → Telegram doesn't allow DM
        print(f"Error sending private joke to {user.id}: {e}")


# ===========================
# SCHEDULED MESSAGE (HOURLY)
# ===========================
async def hourly_message(context: ContextTypes.DEFAULT_TYPE):
    tz = get_tz()
    now = datetime.now(tz)

    if GROUP_CHAT_ID is None:
        print("GROUP_CHAT_ID missing — skipping hourly message.")
        return

    if is_night_time(now):
        print(f"{now} — night time, skip hourly message.")
        return

    try:
        cid = int(GROUP_CHAT_ID)
        await context.bot.send_message(
            chat_id=cid,
            text="Максим, как у тебя дела? Чем занимаешься?",
        )
        print(f"{now} — hourly message sent → {cid}")
    except Exception as e:
        print("Error sending hourly message:", e)


# ===========================
# MAIN APPLICATION
# ===========================
def main():
    if not TOKEN:
        raise RuntimeError("BOT_TOKEN is missing!")

    app = Application.builder().token(TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("chatid", chat_id))
    app.add_handler(CommandHandler("myid", myid))

    # Private echo
    app.add_handler(
        MessageHandler(
            filters.TEXT & filters.ChatType.PRIVATE & ~filters.COMMAND,
            echo_private,
        )
    )

    # Group listener for jokes
    app.add_handler(
        MessageHandler(
            filters.TEXT & filters.ChatType.GROUPS,
            handle_group_message,
        )
    )

    # Scheduling hourly question at HH:15
    tz = get_tz()
    now = datetime.now(tz)
    first_run = compute_next_quarter(now)

    print(
        f"Now: {now} [{TIMEZONE}]. First hourly run: {first_run}. "
        f"Every hour at HH:15, silence 22:00–09:00."
    )

    app.job_queue.run_repeating(
        hourly_message,
        interval=3600,   # 1 hour
        first=first_run,
    )

    print("Bot started with polling, jokes, and hourly scheduler.")
    app.run_polling()


if __name__ == "__main__":
    main()
