import os
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Read bot token and group chat ID from environment variables
TOKEN = os.environ.get("BOT_TOKEN")
GROUP_CHAT_ID = os.environ.get("GROUP_CHAT_ID")  # will set this later in Render


# --------- Basic commands ---------
async def start(update, context: ContextTypes.DEFAULT_TYPE):
    """Reply in private chat or group when user sends /start."""
    await update.message.reply_text(
        "Привет! Я бот Максима. "
        "В группе я буду каждый час спрашивать: "
        "'Максим, как у тебя дела? Чем занимаешься?'"
    )


async def chat_id(update, context: ContextTypes.DEFAULT_TYPE):
    """Send back the chat id so you can configure GROUP_CHAT_ID."""
    cid = update.effective_chat.id
    await update.message.reply_text(f"Chat ID for this chat: `{cid}`", parse_mode="Markdown")


async def echo(update, context: ContextTypes.DEFAULT_TYPE):
    """Simple echo for testing in private chat."""
    text = update.message.text
    await update.message.reply_text(f"Ты написал: {text}")


# --------- Job: hourly message to group ---------
async def hourly_message(context: ContextTypes.DEFAULT_TYPE):
    """Send the hourly message to the configured group."""
    chat_id = GROUP_CHAT_ID

    if not chat_id:
        # Nothing to do if not configured; just log to console
        print("GROUP_CHAT_ID is not set; skipping hourly message.")
        return

    try:
        # convert to int if needed
        chat_id_int = int(chat_id)
        await context.bot.send_message(
            chat_id=chat_id_int,
            text="Максим, как у тебя дела? Чем занимаешься?"
        )
        print("Hourly message sent to", chat_id_int)
    except Exception as e:
        print("Error sending hourly message:", e)


def main():
    # Safety check
    if not TOKEN:
        raise RuntimeError("BOT_TOKEN is not set in environment variables!")

    app = Application.builder().token(TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("chatid", chat_id))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # Schedule hourly job (every 3600 seconds, start immediately)
    job_queue = app.job_queue
    job_queue.run_repeating(hourly_message, interval=3600, first=0)

    print("Bot started and job scheduled...")
    app.run_polling()


if __name__ == "__main__":
    main()
