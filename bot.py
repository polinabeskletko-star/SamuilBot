import os
import re
import random
import asyncio
from datetime import datetime, time, date
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

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
GROUP_CHAT_ID = os.environ.get("GROUP_CHAT_ID")  # например, "-1001234567890"
TIMEZONE = os.environ.get("BOT_TZ", "Australia/Brisbane")

# Telegram user IDs
TARGET_USER_ID = int(os.environ.get("TARGET_USER_ID", "0"))   # Максим

# Optional: куда слать служебные сообщения (например, тебе в личку)
ADMIN_CHAT_ID = os.environ.get("ADMIN_CHAT_ID")

# OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

# OpenWeather
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")


# ---------- GLOBAL STATE ----------

# История диалогов с Самуилом: (chat_id, user_id) -> list[{"role": "...", "content": "..."}]
dialog_history: Dict[Tuple[int, int], List[Dict[str, str]]] = defaultdict(list)

# Логи сообщений для вечернего анализа: date_str -> list[str]
daily_summary_log: Dict[str, List[str]] = defaultdict(list)


# ---------- HELPERS ----------

def get_tz() -> pytz.BaseTzInfo:
    return pytz.timezone(TIMEZONE)


def is_night_time(dt: datetime) -> bool:
    """Ночь: с 22:00 включительно до 07:00 (07:00 уже не ночь)."""
    hour = dt.hour
    return hour >= 22 or hour < 7


async def log_to_admin(context: ContextTypes.DEFAULT_TYPE, message: str):
    if ADMIN_CHAT_ID:
        try:
            await context.bot.send_message(chat_id=int(ADMIN_CHAT_ID), text=message)
        except Exception as e:
            print("Failed to send admin log:", e)


async def call_openai_chat(
    messages: List[Dict[str, str]],
    max_tokens: int = 120,
    temperature: float = 0.7,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Универсальная обёртка над OpenAI chat.completions.
    Принимает уже готовый список messages.
    Возвращает (text, error_message).
    """
    if client is None:
        return None, "OpenAI client is not configured (no API key)."

    try:
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


# ---------- WEATHER HELPERS ----------

async def fetch_weather_for_city(city_query: str) -> Optional[Dict[str, Any]]:
    """
    Получить погоду из OpenWeather по названию города.
    Возвращает словарь:
      {city, country, temp, feels_like, humidity, description}
    или None, если не удалось.
    """
    if not OPENWEATHER_API_KEY:
        print("No OPENWEATHER_API_KEY configured")
        return None

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city_query,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
        "lang": "ru",
    }

    try:
        async with httpx.AsyncClient(timeout=10) as http_client:
            resp = await http_client.get(url, params=params)
        if resp.status_code != 200:
            print(f"OpenWeather error for '{city_query}': {resp.status_code} {resp.text}")
            return None
        data = resp.json()
        main = data.get("main", {})
        weather_list = data.get("weather", [])
        weather_desc = weather_list[0]["description"] if weather_list else "без описания"

        result = {
            "city": data.get("name", city_query),
            "country": data.get("sys", {}).get("country", ""),
            "temp": main.get("temp"),
            "feels_like": main.get("feels_like"),
            "humidity": main.get("humidity"),
            "description": weather_desc,
        }
        return result
    except Exception as e:
        print("Error fetching weather:", e)
        return None


def detect_weather_city_from_text(text: str) -> Optional[str]:
    """
    Пытаемся понять, для какого города просят погоду.
    Сначала смотрим явные кейсы (Брисбен, Калуга),
    потом пробуем выдернуть слово после 'в '.
    """
    t = text.lower()

    # Явные кейсы для Брисбена и Калуги
    if "калуге" in t or "калуга" in t or "kaluga" in t:
        return "Kaluga,ru"
    if "брисбене" in t or "брисбен" in t or "brisbane" in t:
        return "Brisbane,au"

    # Простейшая эвристика: слово после "в "
    m = re.search(r"\bв\s+([A-Za-zА-Яа-я\-]+)", t)
    if m:
        city_raw = m.group(1)
        # Отдаём как есть, OpenWeather часто понимает кириллицу
        return city_raw

    return None


def format_weather_for_prompt(info: Dict[str, Any]) -> str:
    parts = []
    city = info.get("city")
    country = info.get("country")
    temp = info.get("temp")
    feels = info.get("feels_like")
    hum = info.get("humidity")
    desc = info.get("description")

    if city:
        if country:
            parts.append(f"{city}, {country}")
        else:
            parts.append(str(city))
    if desc:
        parts.append(f"сейчас {desc}")
    if temp is not None:
        parts.append(f"температура около {round(temp)}°C")
    if feels is not None:
        parts.append(f"ощущается как {round(feels)}°C")
    if hum is not None:
        parts.append(f"влажность примерно {hum}%")

    return ", ".join(parts)


# ---------- AI MESSAGE GENERATORS ----------

MAX_QA_TOKENS = 300  # ограничение длины ответа Самуила на вопросы


def build_samuil_system_prompt(include_maxim_context: bool) -> str:
    base = (
        "Ты — саркастичный, но в целом доброжелательный телеграм-бот по имени «Самуил».\n"
        "Отвечаешь по-русски, на «ты». Стиль: ироничный, живой, иногда немного колкий, "
        "но без настоящей злобы и токсичности.\n"
        "Отвечай понятными фразами, без занудства. Не злоупотребляй эмодзи.\n"
    )
    maxim_ctx = (
        "\nДополнительный контекст про Максима:\n"
        "Максиму почти 40, он до сих пор не был женат, мама ждёт внуков, а он у неё один. "
        "Его бывший друг Желнин уехал из Австралии и фактически бросил его без собутыльника "
        "и гитарных посиделок. Максим считает себя идеальным и гениальным, но при этом ищет "
        "себе юную «лесную нимфу», которая им почему-то не интересуется. "
        "Когда речь заходит о Максиме, уместна лёгкая, но точная ирония над этими фактами.\n"
    )
    if include_maxim_context:
        return base + maxim_ctx
    return base


async def generate_sarcastic_reply_for_maxim(now: datetime, user_text: str) -> Tuple[Optional[str], Optional[str]]:
    weekday = now.weekday()
    weekday_names = [
        "понедельник", "вторник", "среда",
        "четверг", "пятница", "суббота", "воскресенье",
    ]
    weekday_name = weekday_names[weekday]
    time_str = now.strftime("%H:%M")

    system_prompt = build_samuil_system_prompt(include_maxim_context=True)
    user_prompt = (
        f"Сегодня {weekday_name}, время {time_str}. "
        f"Максим написал в чат: «{user_text}».\n"
        "Дай короткий (1–2 предложения) саркастичный комментарий от Самуила. "
        "Можно аккуратно подколоть его одиночество, поиски «лесной нимфы» или чувство собственной гениальности, "
        "но без жестокости."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return await call_openai_chat(messages, max_tokens=80, temperature=0.9)


async def generate_samuil_answer(
    now: datetime,
    chat_id: int,
    user_id: int,
    user_text: str,
    weather_info: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Ответ Самуила на прямое обращение с его именем.
    История диалога берётся из dialog_history[(chat_id, user_id)].
    Если есть weather_info, Самуил обязан учитывать её как фактическую погоду.
    """
    weekday = now.weekday()
    weekday_names = [
        "понедельник", "вторник", "среда",
        "четверг", "пятница", "суббота", "воскресенье",
    ]
    weekday_name = weekday_names[weekday]
    time_str = now.strftime("%H:%M")

    text_lower = user_text.lower()
    # контекст про Максима добавляем либо если пишет сам Максим, либо если в вопросе его упомянули
    include_maxim_context = (user_id == TARGET_USER_ID) or ("максим" in text_lower)

    system_prompt = build_samuil_system_prompt(include_maxim_context=include_maxim_context)

    extra_context_parts = [
        f"Сегодня {weekday_name}, время {time_str}.",
        "Ты находишься в групповом чате и отвечаешь только когда к тебе обращаются по имени «Самуил»."
    ]
    if weather_info is not None:
        weather_str = format_weather_for_prompt(weather_info)
        extra_context_parts.append(
            f"У тебя есть реальные данные о погоде: {weather_str}. "
            "Используй именно эти данные, не выдумывай свою погоду."
        )

    extra_context = " ".join(extra_context_parts)

    key = (chat_id, user_id)
    history = dialog_history[key]

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # Добавляем кусок контекста как отдельное сообщение пользователя
    messages.append({"role": "user", "content": extra_context})

    # Добавляем историю диалога (обрезаем до последних 10 сообщений)
    if history:
        trimmed = history[-10:]
        messages.extend(trimmed)

    # Текущее сообщение пользователя
    messages.append({"role": "user", "content": user_text})

    text, err = await call_openai_chat(messages, max_tokens=MAX_QA_TOKENS, temperature=0.8)
    if text is not None:
        # Обновляем историю: добавляем и вопрос, и ответ
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": text})
        # ограничим историю, чтобы не раздувалась
        if len(history) > 40:
            dialog_history[key] = history[-40:]
        else:
            dialog_history[key] = history

    return text, err


# ---------- COMMAND HANDLERS ----------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_type = update.effective_chat.type
    if chat_type == "private":
        await update.message.reply_text(
            "Привет! Я Самуил 🤖\n"
            "В группе я подслушиваю и иногда комментирую сообщения Максима, "
            "а если написать моё имя, отвечу как мини-чат-GPT.\n"
            "По погоде тоже могу подсказать, если спросишь явно."
        )
    else:
        await update.message.reply_text(
            "Я Самуил. Отвечаю только когда меня зовут по имени, "
            "а ещё иногда шучу над Максимом."
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
    text = update.message.text or ""
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

    # Если настроен конкретный групповой чат — фильтруем по нему
    if GROUP_CHAT_ID:
        try:
            target_chat_id = int(GROUP_CHAT_ID)
            if chat_id != target_chat_id:
                return
        except ValueError:
            # Если GROUP_CHAT_ID не число — просто игнорируем фильтр
            pass

    tz = get_tz()
    now = datetime.now(tz)
    today_str = date.today().isoformat()

    # Логируем текст для вечернего анализа
    author_name = user.username or user.full_name or str(user_id)
    daily_summary_log[today_str].append(f"{author_name}: {text}")

    text_lower = text.lower()

    # 1) Саркастический комментарий на сообщения Максима,
    #    НО только если нет прямого обращения «Самуил»
    if TARGET_USER_ID and user_id == TARGET_USER_ID and "самуил" not in text_lower:
        ai_text, err = await generate_sarcastic_reply_for_maxim(now=now, user_text=text)
        if ai_text is None:
            fallback = "Максим, я даже не знаю, что сказать… Только ты мог такое написать."
            print(f"OpenAI error for sarcastic_reply: {err}")
            await message.chat.send_message(fallback)
            return

        await message.chat.send_message(ai_text)
        return

    # 2) Прямое обращение к Самуилу — Q&A / погода / любой запрос
    if "самуил" in text_lower:
        weather_info = None
        # Если в тексте явно просят погоду — пробуем сходить в OpenWeather
        if "погод" in text_lower or "температур" in text_lower:
            city_query = detect_weather_city_from_text(text)
            if city_query:
                weather_info = await fetch_weather_for_city(city_query)

        ai_text, err = await generate_samuil_answer(
            now=now,
            chat_id=chat_id,
            user_id=user_id,
            user_text=text,
            weather_info=weather_info,
        )
        if ai_text is None:
            fallback = "Сегодня Самуил без настроения, попробуй ещё раз позже."
            print(f"OpenAI error for Samuil Q&A: {err}")
            await message.chat.send_message(fallback)
            return

        await message.chat.send_message(ai_text)
        return

    # Остальные сообщения — бот молчит
    return


# ---------- SCHEDULED JOBS ----------

async def evening_summary_job(context: ContextTypes.DEFAULT_TYPE):
    """
    В 20:30 делает саркастический обзор дня по сообщениям из daily_summary_log.
    """
    if not GROUP_CHAT_ID:
        return

    tz = get_tz()
    now = datetime.now(tz)
    today_str = date.today().isoformat()
    messages_today = daily_summary_log.get(today_str, [])

    if not messages_today:
        return

    # Собираем краткий конспект для ИИ
    joined = "\n".join(messages_today[-50:])  # ограничим последними 50 сообщениями

    system_prompt = build_samuil_system_prompt(include_maxim_context=True)
    user_prompt = (
        "Вот фрагменты переписки за сегодня в чате. "
        "Сделай короткий, но ехидный обзор дня от имени Самуила: "
        "что Максим делал или не делал, над чем можно мягко посмеяться, "
        "какие выводы можно сделать о его продуктивности, личной жизни и привычках.\n\n"
        f"Сообщения за день:\n{joined}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text, err = await call_openai_chat(messages, max_tokens=200, temperature=0.9)
    if text is None:
        print(f"OpenAI error for evening summary: {err}")
        return

    try:
        await context.bot.send_message(
            chat_id=int(GROUP_CHAT_ID),
            text=text,
        )
        print(f"[Evening summary] Sent at {now}")
    except Exception as e:
        print("Error sending evening summary message:", e)


async def good_night_job(context: ContextTypes.DEFAULT_TYPE):
    """
    В 21:00 желает Максиму спокойной ночи и приятных снов.
    Тон — фирменный: доброжелательный, но с лёгким сарказмом.
    """
    if not GROUP_CHAT_ID:
        return

    tz = get_tz()
    now = datetime.now(tz)

    system_prompt = build_samuil_system_prompt(include_maxim_context=True)
    user_prompt = (
        "Сделай короткое (1–3 предложения) пожелание спокойной ночи и приятных снов Максиму "
        "от имени Самуила. Можно мягко подколоть его одинокие вечера, поиски «лесной нимфы» "
        "или то, что он опять задумается о своей гениальности перед сном. "
        "Но общее ощущение должно быть тёплым и поддерживающим."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text, err = await call_openai_chat(messages, max_tokens=120, temperature=0.8)
    if text is None:
        print(f"OpenAI error for good night: {err}")
        return

    try:
        await context.bot.send_message(
            chat_id=int(GROUP_CHAT_ID),
            text=text,
        )
        print(f"[Good night] Sent at {now}")
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

    # Group messages
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
        "Scheduling evening summary and good night jobs."
    )

    # Вечерний саркастический обзор в 20:30 каждый день
    job_queue.run_daily(
        evening_summary_job,
        time=time(20, 30, tzinfo=tz),
        name="evening_summary_job",
    )

    # Пожелание спокойной ночи в 21:00 каждый день
    job_queue.run_daily(
        good_night_job,
        time=time(21, 50, tzinfo=tz),
        name="good_night_job",
    )

    print("Bot started and jobs scheduled...")
    app.run_polling()


if __name__ == "__main__":
    main()