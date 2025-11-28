import os
import random
import asyncio
from datetime import datetime, time

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

# ==== SETTINGS & ENV ====

TOKEN = os.environ.get("BOT_TOKEN")
GROUP_CHAT_ID = os.environ.get("GROUP_CHAT_ID")  # e.g. "-1001234567890"
TIMEZONE = os.environ.get("BOT_TZ", "Australia/Brisbane")

# Telegram user IDs
TARGET_USER_ID = int(os.environ.get("TARGET_USER_ID", "0"))   # Максим
SUPPORT_USER_ID = int(os.environ.get("SUPPORT_USER_ID", "0")) # Сергей

# Optional: куда слать служебные сообщения (например, тебе в личку)
ADMIN_CHAT_ID = os.environ.get("ADMIN_CHAT_ID")

# OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

client: OpenAI | None = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)


# ---------- HELPERS ----------

def get_tz() -> pytz.BaseTzInfo:
    return pytz.timezone(TIMEZONE)


def is_night_time(dt: datetime) -> bool:
    """
    Ночь: с 22:00 включительно до 07:00 (07:00 уже не ночь).
    """
    hour = dt.hour
    return hour >= 22 or hour < 7


async def log_to_admin(context: ContextTypes.DEFAULT_TYPE, message: str):
    if ADMIN_CHAT_ID:
        try:
            await context.bot.send_message(chat_id=int(ADMIN_CHAT_ID), text=message)
        except Exception as e:
            print("Failed to send admin log:", e)


async def call_openai(system_prompt: str, user_prompt: str,
                      max_tokens: int = 120, temperature: float = 0.7) -> tuple[str | None, str | None]:
    """
    Обёртка над OpenAI. Возвращает (text, error_message).
    """
    if client is None:
        return None, "OpenAI client is not configured (no API key)."

    try:
        # В отдельном потоке, чтобы не блокировать event loop
        resp = await asyncio.to_thread(
            client.chat.completions.create,
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = resp.choices[0].message.content.strip()
        return text, None
    except Exception as e:
        err = f"Error calling OpenAI: {e}"
        print(err)
        return None, err


async def generate_message_for_kind(kind: str,
                                    now: datetime,
                                    user_text: str | None = None) -> tuple[str | None, str | None]:
    """
    kind:
      - "sarcastic_reply"  — ответ Максиму
      - "support_for_maxim" — поддержка от имени бота на сообщения Сергея
      - "weekend_hourly"   — часовой вопрос по выходным
      - "weekday_morning"  — утреннее сообщение по будням
    """
    weekday = now.weekday()  # 0=Mon ... 6=Sun
    weekday_names = ["понедельник", "вторник", "среда", "четверг",
                     "пятница", "суббота", "воскресенье"]
    weekday_name = weekday_names[weekday]
    time_str = now.strftime("%H:%M")

    if kind == "sarcastic_reply":
        system_prompt = (
            "Ты дружелюбный, но слегка саркастичный бот-друг по имени 'Друг Максима'. "
            "Ты пишешь по-русски, на 'ты', коротко (1–2 предложения). "
            "Мягко подкалывай Максима, но без грубости и токсичности. "
            "Не используй эмодзи в каждом сообщении, максимум один, и не всегда."
        )
        user_prompt = (
            f"Сегодня {weekday_name}, время {time_str}. "
            f"Максим написал в чат: «{user_text}».\n"
            "Ответь коротко, с лёгкой иронией. Не повторяй дословно текст Максима. "
            "Сообщение должно быть самостоятельным, а не выглядеть как явный ответ."
        )
        return await call_openai(system_prompt, user_prompt, max_tokens=80, temperature=0.8)

    if kind == "support_for_maxim":
        system_prompt = (
            "Ты бот-поддержка Максима. Ты видишь сообщения от другого человека, "
            "который его подбадривает. Твоя задача — добавить ещё одну короткую, "
            "искреннюю, но не приторную поддержку для Максима. Пиши по-русски, на 'ты'. "
            "1 короткое предложение, максимум два. Не будь слишком льстивым, "
            "избегай громких слов типа 'невероятный', 'величайший' и т.п. "
            "Сообщение должно быть самостоятельным высказыванием, не ответом этому человеку. "
            "Обязательно упоминай Максима по имени хотя бы один раз."
        )
        user_prompt = (
            f"Сегодня {weekday_name}, время {time_str}. "
            f"Другой человек написал в чат слова поддержки Максиму: «{user_text}».\n"
            "Сформулируй от себя ещё одну естественную, живую поддержку для Максима."
        )
        return await call_openai(system_prompt, user_prompt, max_tokens=60, temperature=0.7)

    if kind == "weekend_hourly":
        system_prompt = (
            "Ты бот-друг Максима в Telegram-чате. "
            "По выходным ты примерно раз в час задаёшь Максиму вопрос, как у него дела "
            "и чем он занят. Пиши по-русски, на 'ты'. "
            "Коротко: 1–2 предложения. Можно иногда чуть иронично, но по-доброму. "
            "Не повторяй каждый раз одну и ту же формулировку. "
            "Не злоупотребляй эмодзи — максимум один, и не в каждом сообщении."
        )
        user_prompt = (
            f"Сейчас {weekday_name}, {time_str}. "
            "Придумай очередной вопрос или небольшое обращение к Максиму, "
            "которое поможет ему почувствовать внимание и немного улыбнуться."
        )
        return await call_openai(system_prompt, user_prompt, max_tokens=80, temperature=0.8)

    if kind == "weekday_morning":
        system_prompt = (
            "Ты бот-друг Максима в рабочем чате. "
            "По будням в 7 утра ты желаешь Максиму доброго утра и хорошего рабочего дня. "
            "Пиши по-русски, на 'ты', 1–2 предложения. "
            "Лёгкий, доброжелательный тон, можно с лёгким юмором. "
            "Упоминай, что впереди рабочий день. Эмодзи можно, но не обязательно."
        )
        user_prompt = (
            f"Сегодня {weekday_name}, время {time_str}. "
            "Сделай короткое утреннее сообщение для Максима: поздоровайся, "
            "пожелай хорошего рабочего дня и намекни, что ты рядом и будешь за ним наблюдать в чате."
        )
        return await call_openai(system_prompt, user_prompt, max_tokens=80, temperature=0.7)

    return None, "Unknown message kind"


# ---------- COMMAND HANDLERS ----------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_type = update.effective_chat.type
    if chat_type == "private":
        await update.message.reply_text(
            "Привет! Я Друг Максима 🤖\n"
            "В группе я буду:\n"
            "• По будням в 7:00 желать Максиму доброго утра и хорошего рабочего дня.\n"
            "• По выходным писать ему примерно раз в час в случайное время.\n"
            "Ночью с 22:00 до 7:00 я молчу 😴"
        )
    else:
        await update.message.reply_text(
            "Я здесь, чтобы поддерживать Максима:\n"
            "• Будни: сообщение в 7:00.\n"
            "• Выходные: раз в час в случайную минуту.\n"
            "Ночью с 22:00 до 7:00 я не беспокою."
        )


async def chat_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    await update.message.reply_text(
        f"Chat ID for this chat: `{cid}`",
        parse_mode="Markdown",
    )


async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await update.message.reply_text(
        f"Your user ID: `{user.id}`\nUsername: @{user.username}",
        parse_mode="Markdown",
    )


async def echo_private(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Echo только в личке, в группах молчим."""
    if update.effective_chat.type != "private":
        return
    text = update.message.text
    await update.message.reply_text(f"Ты написал: {text}")


# ---------- GROUP MESSAGE HANDLER ----------

async def handle_group_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if message is None:
        return

    chat = message.chat
    user = message.from_user
    text = message.text or ""

    chat_id = chat.id
    user_id = user.id

    print(
        f"DEBUG UPDATE: chat_id={chat_id} chat_type={chat.type} "
        f"user_id={user_id} user_name={user.username} text='{text}'"
    )

    # Если это не целевой групповой чат, ничего не делаем
    if GROUP_CHAT_ID and int(GROUP_CHAT_ID) != chat_id:
        return

    tz = get_tz()
    now = datetime.now(tz)

    # Сообщения Максима — саркастичный ответ
    if TARGET_USER_ID and user_id == TARGET_USER_ID:
        ai_text, err = await generate_message_for_kind(
            "sarcastic_reply", now=now, user_text=text
        )
        if ai_text is None:
            fallback = "Максим, я даже не знаю, что сказать… Ты сам понял, что написал? 😉"
            print(f"OpenAI error for sarcastic_reply: {err}")
            await message.chat.send_message(fallback)
            return

        await message.chat.send_message(ai_text)
        return

    # Сообщения Сергея — дополнительная поддержка Максима
    if SUPPORT_USER_ID and user_id == SUPPORT_USER_ID:
        ai_text, err = await generate_message_for_kind(
            "support_for_maxim", now=now, user_text=text
        )
        if ai_text is None:
            fallback = "Максим, кажется, вселенная сегодня явно за тебя."
            print(f"OpenAI error for support_for_maxim: {err}")
            await message.chat.send_message(fallback)
            return

        await message.chat.send_message(ai_text)
        return

    # Остальные пользователи — бот молчит (в группе)
    return


# ---------- SCHEDULED JOBS ----------

async def weekend_random_hourly_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Запускается каждую минуту.
    По выходным раз в час выбирает случайную минуту и в неё шлёт сообщение Максиму.
    """
    if not GROUP_CHAT_ID:
        return

    tz = get_tz()
    now = datetime.now(tz)

    weekday = now.weekday()  # 0=Mon ... 6=Sun
    if weekday < 5:
        # Будни — этим джобом не занимаемся
        return

    # Ночной режим
    if is_night_time(now):
        return

    job = context.job
    if job.data is None:
        job.data = {}

    data = job.data
    current_hour = now.hour
    last_hour = data.get("last_hour")
    target_minute = data.get("target_minute")
    sent_this_hour = data.get("sent_this_hour", False)

    # Новый час — планируем новую случайную минуту и сбрасываем флаг
    if last_hour is None or current_hour != last_hour:
        target_minute = random.randint(0, 59)
        sent_this_hour = False
        data["last_hour"] = current_hour
        data["target_minute"] = target_minute
        data["sent_this_hour"] = sent_this_hour
        print(f"[Weekend scheduler] New hour {current_hour}, planned minute {target_minute}")

    # Если ещё не отправляли в этом часе и наступила нужная минута — шлём
    if not sent_this_hour and now.minute == target_minute:
        text, err = await generate_message_for_kind(
            "weekend_hourly", now=now
        )
        if text is None:
            text = "Максим, как у тебя дела? Чем сейчас занимаешься?"
            print(f"OpenAI error for weekend_hourly: {err}")

        try:
            await context.bot.send_message(
                chat_id=int(GROUP_CHAT_ID),
                text=text,
            )
            data["sent_this_hour"] = True
            print(f"[Weekend scheduler] Sent hourly message at {now}")
        except Exception as e:
            print("Error sending weekend hourly message:", e)

    job.data = data


async def weekday_morning_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Запускается в 7:00 по будням.
    """
    if not GROUP_CHAT_ID:
        return

    tz = get_tz()
    now = datetime.now(tz)

    weekday = now.weekday()
    if weekday >= 5:
        # На всякий случай: по выходным это сообщение не нужно
        return

    text, err = await generate_message_for_kind(
        "weekday_morning", now=now
    )
    if text is None:
        text = "Доброе утро, Максим! Удачи сегодня на работе — я слежу за тобой из чата. 😉"
        print(f"OpenAI error for weekday_morning: {err}")

    try:
        await context.bot.send_message(
            chat_id=int(GROUP_CHAT_ID),
            text=text,
        )
        print(f"[Weekday morning] Sent morning message at {now}")
    except Exception as e:
        print("Error sending weekday morning message:", e)


# ---------- MAIN APP ----------

def main():
    if not TOKEN:
        raise RuntimeError("BOT_TOKEN is not set in environment variables!")

    app = Application.builder().token(TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("chatid", chat_id))
    app.add_handler(CommandHandler("whoami", whoami))

    # Echo only in private chats
    app.add_handler(
        MessageHandler(
            filters.TEXT & filters.ChatType.PRIVATE & ~filters.COMMAND,
            echo_private,
        )
    )

    # Group messages in target chat
    app.add_handler(
        MessageHandler(
            filters.TEXT & filters.ChatType.GROUPS & ~filters.COMMAND,
            handle_group_message,
        )
    )

    # JobQueue scheduling
    job_queue = app.job_queue
    tz = get_tz()
    now = datetime.now(tz)

    print(
        f"Local time now: {now} [{TIMEZONE}]. "
        "Scheduling weekday morning and weekend hourly jobs."
    )

    # 1) Будние утренние сообщения в 7:00 (пн–пт)
    # days=(0,1,2,3,4) — понедельник-пятница
    job_queue.run_daily(
        weekday_morning_job,
        time=time(7, 0, tzinfo=tz),
        days=(0, 1, 2, 3, 4),
        name="weekday_morning_job",
    )

    # 2) Выходные: джоба раз в минуту, внутри — логика случайной минуты
    job_queue.run_repeating(
        weekend_random_hourly_job,
        interval=60,          # каждую минуту
        first=0,              # сразу
        name="weekend_random_hourly_job",
        data={},              # для хранения состояния по часам
    )

    print("Bot started and jobs scheduled...")
    app.run_polling()


if __name__ == "__main__":
    main()