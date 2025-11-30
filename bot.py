import os
import random
import asyncio
from datetime import datetime, date, time as dtime
from typing import Optional, Tuple, List, Dict

import pytz
import httpx
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

client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)


# ---------- HELPERS: TIME & TZ ----------

def get_tz() -> pytz.BaseTzInfo:
    return pytz.timezone(TIMEZONE)


def is_night_time(dt: datetime) -> bool:
    """
    Ночь: с 22:00 включительно до 07:00 (07:00 уже не ночь).
    """
    hour = dt.hour
    return hour >= 22 or hour < 7


# ---------- HELPERS: WEATHER ----------

BRISBANE_LAT, BRISBANE_LON = -27.47, 153.03
KALUGA_LAT, KALUGA_LON = 54.51, 36.27


async def fetch_weather(
    lat: float,
    lon: float,
    tz_str: str,
) -> Optional[dict]:
    """
    Простая обёртка над Open-Meteo (без API ключа).
    Возвращает dict с текущей и дневной температурой либо None при ошибке.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,weather_code"
        "&daily=temperature_2m_max,temperature_2m_min"
        f"&timezone={tz_str}"
    )

    try:
        async with httpx.AsyncClient(timeout=10) as client_http:
            resp = await client_http.get(url)
            resp.raise_for_status()
            data = resp.json()
            return data
    except Exception as e:
        print(f"Weather fetch error: {e}")
        return None


def format_brisbane_weather_short(data: Optional[dict]) -> str:
    if not data:
        return "Погода в Брисбене сегодня как жизнь Максима — непредсказуемая."
    try:
        current = data["current"]
        daily = data["daily"]
        temp_now = current["temperature_2m"]
        tmin = daily["temperature_2m_min"][0]
        tmax = daily["temperature_2m_max"][0]
        return (
            f"В Брисбене сейчас около {round(temp_now)}°C, "
            f"днём от {round(tmin)}°C до {round(tmax)}°C."
        )
    except Exception as e:
        print("Weather format error:", e)
        return "Погода в Брисбене сегодня странная, как отчёты по KPI."


def format_weather_compare(
    brisbane: Optional[dict],
    kaluga: Optional[dict],
) -> str:
    if not brisbane and not kaluga:
        return "Даже погода отказалась обновляться. Идеальный день для философии, Максим."

    def safe_extract(data, name):
        if not data:
            return None, None, None
        try:
            current = data["current"]
            daily = data["daily"]
            temp_now = current["temperature_2m"]
            tmin = daily["temperature_2m_min"][0]
            tmax = daily["temperature_2m_max"][0]
            return temp_now, tmin, tmax
        except Exception as e:
            print(f"Weather parse error for {name}:", e)
            return None, None, None

    br_now, br_min, br_max = safe_extract(brisbane, "Brisbane")
    ka_now, ka_min, ka_max = safe_extract(kaluga, "Kaluga")

    if br_now is None and ka_now is None:
        return "Погода молчит и в Брисбене, и в Калуге. Видимо, вселенная взяла выходной."

    parts = []
    if br_now is not None:
        parts.append(
            f"В Брисбене сейчас около {round(br_now)}°C"
            f" (днём {round(br_min)}–{round(br_max)}°C)"
        )
    if ka_now is not None:
        parts.append(
            f"В Калуге сейчас около {round(ka_now)}°C"
            f" (днём {round(ka_min)}–{round(ka_max)}°C)"
        )

    text = " | ".join(parts)

    # Немного сарказма в конце
    if br_now is not None and ka_now is not None:
        if br_now > ka_now + 10:
            text += " — Максим, у тебя климатический читы включены."
        elif ka_now > br_now + 10:
            text += " — Похоже, Калуга сегодня решила погреться за двоих."
        else:
            text += " — В целом шансы выжить там и там примерно равные."

    return text


# ---------- HELPERS: LOGGING & OPENAI ----------

async def log_to_admin(context: ContextTypes.DEFAULT_TYPE, message: str):
    if ADMIN_CHAT_ID:
        try:
            await context.bot.send_message(chat_id=int(ADMIN_CHAT_ID), text=message)
        except Exception as e:
            print("Failed to send admin log:", e)


async def call_openai(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 120,
    temperature: float = 0.7,
    history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Обёртка над OpenAI. Возвращает (text, error_message).

    history: список сообщений вида {"role": "user"/"assistant", "content": "..."}
    который добавляется перед текущим user_prompt (используется для Самуила).
    """
    if client is None:
        return None, "OpenAI client is not configured (no API key)."

    try:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        resp = await asyncio.to_thread(
            client.chat.completions.create,
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = resp.choices[0].message.content.strip()
        return text, None
    except Exception as e:
        err = f"Error calling OpenAI: {e}"
        print(err)
        return None, err


async def generate_message_for_kind(
    kind: str,
    now: datetime,
    user_text: Optional[str] = None,
    daily_messages: Optional[list] = None,
    weather_brisbane: Optional[dict] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    kind:
      - "sarcastic_reply"    — ответ Максиму
      - "support_for_maxim"  — поддержка Максима
      - "weekend_regular"    — регулярное выходное с упоминанием погоды
      - "weekday_morning"    — утреннее сообщение в будни с погодой
      - "evening_summary"    — анализ дня
      - "weather_compare"    — сравнение погоды Бризбен / Калуга
      - "good_night"         — спокойной ночи
      - "samuil_qa"          — ответ Самуила на прямой вопрос (слово «Самуил»)
    """
    weekday = now.weekday()  # 0=Mon ... 6=Sun
    weekday_names = [
        "понедельник",
        "вторник",
        "среда",
        "четверг",
        "пятница",
        "суббота",
        "воскресенье",
    ]
    weekday_name = weekday_names[weekday]
    time_str = now.strftime("%H:%M")

    # --- Сарказм Максиму ---
    if kind == "sarcastic_reply":
        system_prompt = (
            "Ты дружелюбный, но максимально саркастичный бот-друг по имени 'Самуил'. "
            "Ты пишешь по-русски, на 'ты', коротко (1–2 предложения). "
            "Мягко подкалывай Максима, но без откровенной грубости или оскорблений. "
            "Не повторяй дословно текст Максима. Можно использовать лёгкий чёрный юмор."
        )
        user_prompt = (
            f"Сегодня {weekday_name}, время {time_str}. "
            f"Максим написал в чат: «{user_text}».\n"
            "Ответь коротко и саркастично, будто ты старый друг, "
            "который уже ничему не удивляется. Не пиши, что отвечаешь на это сообщение, "
            "просто сделай самостоятельное утверждение."
        )
        return await call_openai(system_prompt, user_prompt, max_tokens=80, temperature=0.9)

    # --- Поддержка Максима (по сообщениям Сергея) ---
    if kind == "support_for_maxim":
        system_prompt = (
            "Ты бот-поддержка Максима по имени 'Самуил'. "
            "Ты видишь сообщения от другого человека, который его подбадривает. "
            "Твоя задача — добавить ещё одну короткую, искреннюю, но не приторную поддержку. "
            "Пиши по-русски, на 'ты'. 1 короткое предложение, максимум два. "
            "Не используй громкие слова типа 'величайший', 'невероятный'. "
            "Сообщение должно быть самостоятельным высказыванием, не ответом этому человеку. "
            "Обязательно упоминай Максима по имени хотя бы один раз."
        )
        user_prompt = (
            f"Сегодня {weekday_name}, время {time_str}. "
            f"Другой человек написал в чат слова поддержки Максиму: «{user_text}».\n"
            "Сформулируй от себя ещё одну естественную, живую поддержку для Максима."
        )
        return await call_openai(system_prompt, user_prompt, max_tokens=60, temperature=0.7)

    # --- Регулярное выходное сообщение с погодой ---
    if kind == "weekend_regular":
        weather_text = format_brisbane_weather_short(weather_brisbane)
        system_prompt = (
            "Ты бот-друг Максима в Telegram-чате по имени 'Самуил'. "
            "По выходным ты несколько раз в день пишешь Максиму, спрашиваешь как дела "
            "и слегка его подшучиваешь. Пиши по-русски, на 'ты', 1–2 предложения. "
            "Тон лёгкий, саркастичный, но доброжелательный. "
            "В тексте можно упомянуть погоду, но не слишком сухо."
        )
        user_prompt = (
            f"Сейчас {weekday_name}, {time_str}. "
            f"Краткая сводка погоды: {weather_text}\n"
            "Придумай смешное короткое сообщение для Максима: спроси как он, "
            "упомяни, что ты в курсе погоды, и слегка подтруни над ним."
        )
        return await call_openai(system_prompt, user_prompt, max_tokens=90, temperature=0.9)

    # --- Утреннее сообщение в будни с погодой ---
    if kind == "weekday_morning":
        weather_text = format_brisbane_weather_short(weather_brisbane)
        system_prompt = (
            "Ты бот-друг Максима 'Самуил'. "
            "По будням в 7 утра ты желаешь Максиму доброго утра и хорошего рабочего дня. "
            "Пиши по-русски, на 'ты', 1–2 предложения. "
            "Тон доброжелательный, с лёгким юмором и лёгкой иронией. "
            "В сообщении обязательно кратко упомяни погоду на день."
        )
        user_prompt = (
            f"Сегодня {weekday_name}, время {time_str}. "
            f"Сводка погоды: {weather_text}\n"
            "Сделай короткое утреннее сообщение для Максима: поздоровайся, "
            "пожелай хорошего рабочего дня, вмонтируй погоду в текст и немного пошути."
        )
        return await call_openai(system_prompt, user_prompt, max_tokens=90, temperature=0.8)

    # --- Вечерний анализ дня ---
    if kind == "evening_summary":
        system_prompt = (
            "Ты саркастичный, но не злой Telegram-бот по имени 'Самуил'. "
            "Твоя задача — сделать короткий обзор дня Максима по логам сообщений из чата. "
            "Пиши по-русски, на 'ты'. 2–4 предложения. "
            "Используй иронию, подчёркивай забавные моменты, но не переходи в откровенные оскорбления. "
            "Если сообщений мало, тоже сделай саркастичный вывод."
        )
        logs_text = "\n".join(daily_messages or [])
        if not logs_text:
            logs_text = "Сегодня в чате почти ничего не происходило."

        user_prompt = (
            f"Сегодня {weekday_name}, время {time_str}. "
            "Вот список важных сообщений из чата за день (может быть пустым):\n"
            f"{logs_text}\n\n"
            "Сделай краткий саркастичный, но доброжелательный обзор дня Максима."
        )
        return await call_openai(system_prompt, user_prompt, max_tokens=160, temperature=0.8)

    # --- Сравнение погоды Бризбен / Калуга ---
    if kind == "weather_compare":
        # Здесь сам текст уже формируется через format_weather_compare,
        # поэтому просто вернём код ошибки, чтобы не использовать этот путь.
        return None, "weather_compare_should_be_built_outside"

    # --- Спокойной ночи ---
    if kind == "good_night":
        system_prompt = (
            "Ты бот-друг Максима 'Самуил'. "
            "В 21:00 ты желаешь ему спокойной ночи и приятных снов. "
            "Пиши по-русски, на 'ты', 1–2 предложения. "
            "Можно чуть подшутить насчёт его дня или планов на завтра, но мягко."
        )
        user_prompt = (
            f"Сегодня {weekday_name}, время {time_str}. "
            "Придумай короткое пожелание спокойной ночи Максиму, "
            "с лёгкой иронией и намёком, что ты будешь ждать его завтра в чате."
        )
        return await call_openai(system_prompt, user_prompt, max_tokens=80, temperature=0.8)

    # --- Q&A по имени 'Самуил' с учётом истории ---
    if kind == "samuil_qa":
        system_prompt = (
            "Ты умный, остроумный и слегка саркастичный помощник по имени 'Самуил'. "
            "Ты отвечаешь на вопросы пользователей в Telegram-чате. "
            "Пиши по-русски, на 'ты', давай полезные и по возможности точные ответы. "
            "Можешь немного подшучивать, но не будь откровенно грубым. "
            "Отвечай по сути вопроса. "
            "История переписки, которая идёт перед последним сообщением, — это ваш контекст разговора."
        )
        user_prompt = (
            f"Сегодня {weekday_name}, время {time_str}. "
            f"Новое сообщение пользователя в чате, где упомянули тебя по имени:\n"
            f"«{user_text}».\n\n"
            "Считай это продолжением разговора и ответь по сути. "
            "2–5 предложений, не слишком длинно."
        )
        return await call_openai(
            system_prompt,
            user_prompt,
            max_tokens=220,
            temperature=0.8,
            history=history,
        )

    return None, "Unknown message kind"


# ---------- COMMAND HANDLERS ----------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_type = update.effective_chat.type
    if chat_type == "private":
        await update.message.reply_text(
            "Привет! Я Самуил 🤖\n"
            "В группе я:\n"
            "• По будням в 7:00 желаю Максиму доброго утра с учётом погоды в Брисбене.\n"
            "• По выходным несколько раз в день напоминаю о себе сообщениями с вопросом и шутками.\n"
            "• В 20:30 даю саркастичный обзор дня.\n"
            "• В 21:00 желаю спокойной ночи.\n"
            "• Если в сообщении есть слово «Самуил» — считаю это вопросом и отвечаю как мини-ChatGPT с памятью.\n"
            "Ночью с 22:00 до 7:00 я молчу 😴"
        )
    else:
        await update.message.reply_text(
            "Я Самуил. В этом чате я:\n"
            "• Подшучиваю над Максимом,\n"
            "• Добавляю поддержку Максиму, когда его поддерживает Сергей,\n"
            "• Пишу регулярные сообщения с учётом погоды,\n"
            "• Делаю вечерний обзор дня и желаю спокойной ночи,\n"
            "• И отвечаю на вопросы, где есть слово «Самуил», помня контекст беседы."
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

    # Логи в консоль
    print(
        f"DEBUG UPDATE: chat_id={chat_id} chat_type={chat.type} "
        f"user_id={user_id} user_name={user.username} text='{text}'"
    )

    # Если это не целевой групповой чат, ничего не делаем
    if GROUP_CHAT_ID and int(GROUP_CHAT_ID) != chat_id:
        return

    tz = get_tz()
    now = datetime.now(tz)

    # Копим сообщения для вечернего анализа (простая реализация в памяти)
    today_str = date.today().isoformat()
    bot_data = context.bot_data
    key_daily = f"daily_messages_{today_str}"
    msgs_list = bot_data.get(key_daily, [])
    msgs_list.append(f"{user.username or user.full_name}: {text}")
    bot_data[key_daily] = msgs_list

    text_lower = text.lower()

    # Ключ для истории диалога с Самуилом в этом чате
    history_key = f"samuil_history_{chat_id}"
    history: List[Dict[str, str]] = bot_data.get(history_key, [])

    # --- 1) Вопрос к Самуилу по ключевому слову (имеет приоритет над остальным) ---
    if "самуил" in text_lower:
        ai_text, err = await generate_message_for_kind(
            "samuil_qa",
            now=now,
            user_text=text,
            history=history,
        )
        if ai_text is None:
            fallback = "Я услышал, что ты меня звал, но у меня сейчас экзистенциальный тайм-аут."
            print(f"OpenAI error for samuil_qa: {err}")
            await message.chat.send_message(fallback)
            return

        # Обновляем историю: добавляем реплику пользователя и ответ Самуила
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": ai_text})

        # Ограничиваем историю, чтобы не раздувалась бесконечно
        if len(history) > 20:
            history = history[-20:]

        bot_data[history_key] = history

        await message.chat.send_message(ai_text)
        return

    # --- 2) Сообщения Максима — саркастичный ответ ---
    if TARGET_USER_ID and user_id == TARGET_USER_ID:
        ai_text, err = await generate_message_for_kind(
            "sarcastic_reply",
            now=now,
            user_text=text,
        )
        if ai_text is None:
            fallback = "Максим, я даже не знаю, что сказать… Ты сам понял, что написал? 😉"
            print(f"OpenAI error for sarcastic_reply: {err}")
            await message.chat.send_message(fallback)
            return

        await message.chat.send_message(ai_text)
        return

    # --- 3) Сообщения Сергея — поддержка Максима, только если есть 'максим' ---
    if SUPPORT_USER_ID and user_id == SUPPORT_USER_ID:
        if "максим" in text_lower:
            ai_text, err = await generate_message_for_kind(
                "support_for_maxim",
                now=now,
                user_text=text,
            )
            if ai_text is None:
                fallback = "Максим, у тебя тут сильная группа поддержки, не подведи."
                print(f"OpenAI error for support_for_maxim: {err}")
                await message.chat.send_message(fallback)
                return

            await message.chat.send_message(ai_text)
        return

    # Остальные пользователи — бот молчит (если не упомянули Самуила)
    return


# ---------- SCHEDULED JOBS ----------

async def weekend_regular_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Раз в заданное время по выходным — сообщение Максиму с упоминанием погоды.
    """
    if not GROUP_CHAT_ID:
        return

    tz = get_tz()
    now = datetime.now(tz)
    weekday = now.weekday()  # 0=Mon ... 6=Sun

    if weekday < 5:
        return  # Только суббота/воскресенье

    if is_night_time(now):
        return

    weather_data = await fetch_weather(
        BRISBANE_LAT,
        BRISBANE_LON,
        TIMEZONE,
    )

    text, err = await generate_message_for_kind(
        "weekend_regular",
        now=now,
        weather_brisbane=weather_data,
    )
    if text is None:
        text = "Максим, как у тебя дела? Погода в Брисбене живёт своей жизнью, как и ты."
        print(f"OpenAI error for weekend_regular: {err}")

    try:
        await context.bot.send_message(
            chat_id=int(GROUP_CHAT_ID),
            text=text,
        )
        print(f"[Weekend regular] Sent weekend message at {now}")
    except Exception as e:
        print("Error sending weekend regular message:", e)


async def weekday_morning_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Будние утренние сообщения в 7:00.
    """
    if not GROUP_CHAT_ID:
        return

    tz = get_tz()
    now = datetime.now(tz)
    weekday = now.weekday()

    if weekday >= 5:
        return  # На всякий случай

    weather_data = await fetch_weather(
        BRISBANE_LAT,
        BRISBANE_LON,
        TIMEZONE,
    )

    text, err = await generate_message_for_kind(
        "weekday_morning",
        now=now,
        weather_brisbane=weather_data,
    )
    if text is None:
        text = (
            "Доброе утро, Максим! Погода там за окном что-то показывает, "
            "а тебе всё равно на работу. Держись. 😉"
        )
        print(f"OpenAI error for weekday_morning: {err}")

    try:
        await context.bot.send_message(
            chat_id=int(GROUP_CHAT_ID),
            text=text,
        )
        print(f"[Weekday morning] Sent morning message at {now}")
    except Exception as e:
        print("Error sending weekday morning message:", e)


async def evening_summary_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Вечерний анализ дня в 20:30.
    """
    if not GROUP_CHAT_ID:
        return

    tz = get_tz()
    now = datetime.now(tz)
    today_str = date.today().isoformat()

    bot_data = context.bot_data
    key = f"daily_messages_{today_str}"
    daily_messages = bot_data.get(key, [])

    text, err = await generate_message_for_kind(
        "evening_summary",
        now=now,
        daily_messages=daily_messages,
    )
    if text is None:
        text = "Сегодня в чате было так тихо, что я почти поверил в продуктивность."
        print(f"OpenAI error for evening_summary: {err}")

    try:
        await context.bot.send_message(
            chat_id=int(GROUP_CHAT_ID),
            text=text,
        )
        print(f"[Evening summary] Sent summary at {now}")
    except Exception as e:
        print("Error sending evening summary message:", e)

    # После отчёта можно очистить логи за день
    bot_data[key] = []


async def weather_compare_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Хотя бы раз в день сравнение погоды в Брисбене и Калуге.
    Пусть будет в 12:00.
    """
    if not GROUP_CHAT_ID:
        return

    br_data = await fetch_weather(
        BRISBANE_LAT,
        BRISBANE_LON,
        TIMEZONE,
    )
    # Калуга в таймзоне Москвы
    ka_data = await fetch_weather(
        KALUGA_LAT,
        KALUGA_LON,
        "Europe/Moscow",
    )

    compare_text = format_weather_compare(br_data, ka_data)

    try:
        await context.bot.send_message(
            chat_id=int(GROUP_CHAT_ID),
            text=compare_text,
        )
        print("[Weather compare] Sent Brisbane vs Kaluga weather message")
    except Exception as e:
        print("Error sending weather compare message:", e)


async def good_night_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Пожелание спокойной ночи в 21:00.
    """
    if not GROUP_CHAT_ID:
        return

    tz = get_tz()
    now = datetime.now(tz)

    text, err = await generate_message_for_kind(
        "good_night",
        now=now,
    )
    if text is None:
        text = "Спокойной ночи, Максим. Постарайся не думать о работе хотя бы во сне."
        print(f"OpenAI error for good_night: {err}")

    try:
        await context.bot.send_message(
            chat_id=int(GROUP_CHAT_ID),
            text=text,
        )
        print(f"[Good night] Sent good night message at {now}")
    except Exception as e:
        print("Error sending good night message:", e)


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
        "Scheduling weekday morning, weekend regular, weather compare, evening summary and good night jobs."
    )

    # 1) Будние утренние сообщения в 7:00 (пн–пт)
    job_queue.run_daily(
        weekday_morning_job,
        time=dtime(7, 0, tzinfo=tz),
        days=(0, 1, 2, 3, 4),
        name="weekday_morning_job",
    )

    # 2) Выходные: сообщения несколько раз в день — в 9, 12, 15, 18
    for hour in (9, 12, 15, 18):
        job_queue.run_daily(
            weekend_regular_job,
            time=dtime(hour, 0, tzinfo=tz),
            days=(5, 6),  # суббота, воскресенье
            name=f"weekend_regular_{hour}",
        )

    # 3) Сравнение погоды Бризбен / Калуга в 12:00 каждый день
    job_queue.run_daily(
        weather_compare_job,
        time=dtime(12, 0, tzinfo=tz),
        days=(0, 1, 2, 3, 4, 5, 6),
        name="weather_compare_job",
    )

    # 4) Вечерний анализ дня в 20:30
    job_queue.run_daily(
        evening_summary_job,
        time=dtime(20, 30, tzinfo=tz),
        days=(0, 1, 2, 3, 4, 5, 6),
        name="evening_summary_job",
    )

    # 5) Спокойной ночи в 21:00
    job_queue.run_daily(
        good_night_job,
        time=dtime(21, 0, tzinfo=tz),
        days=(0, 1, 2, 3, 4, 5, 6),
        name="good_night_job",
    )

    print("Bot started and jobs scheduled...")
    app.run_polling()


if __name__ == "__main__":
    main()