import os
import re
import random
import asyncio
from datetime import datetime, time, date, timedelta
from collections import defaultdict, deque
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
OPENAI_IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")

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

# Флаг для отслеживания, были ли уже добавлены задачи (в рамках процесса)
_jobs_scheduled = False

# Дедуп отправки плановых сообщений (в рамках процесса)
# job_name -> datetime last_sent_at (tz-aware)
_last_scheduled_sent_at: Dict[str, datetime] = {}
# job_name -> deque последних текстов
_last_scheduled_texts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))

# Для разнообразия ответов Максиму: хранить последние ответы
_last_maxim_replies: deque = deque(maxlen=8)


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
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return None, "Empty response from OpenAI."
        return text, None
    except Exception as e:
        err = f"Error calling OpenAI: {e}"
        print(err)
        return None, err


async def generate_image_from_prompt(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Генерация картинки через OpenAI Images по текстовому запросу.
    Возвращает (image_url, error_message).
    """
    if client is None:
        return None, "OpenAI client is not configured (no API key)."

    try:
        resp = await asyncio.to_thread(
            client.images.generate,
            model=OPENAI_IMAGE_MODEL,
            prompt=prompt,
            n=1,
            size="1024x1024",
        )
        image_url = resp.data[0].url
        return image_url, None
    except Exception as e:
        err = f"Error calling OpenAI Images: {e}"
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

    if "калуге" in t or "калуга" in t or "kaluga" in t:
        return "Kaluga,ru"
    if "брисбене" in t or "брисбен" in t or "brisbane" in t:
        return "Brisbane,au"

    m = re.search(r"\bв\s+([A-Za-zА-Яа-я\-]+)", t)
    if m:
        city_raw = m.group(1)
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

# Сократили лимиты, чтобы ответы стали короче
MAX_QA_TOKENS = 160  # ответы Самуила при прямом обращении
MAX_MAXIM_REPLY_TOKENS = 70  # ответы на Максима (коротко)

def get_time_context(time_str: str, hour: int) -> str:
    """Генерирует контекстное описание времени для промптов."""
    if hour < 6:
        return "Ночь."
    elif hour < 12:
        return "Утро."
    elif hour < 17:
        return "День."
    elif hour < 22:
        return "Вечер."
    else:
        return "Поздний вечер."


def build_samuil_system_prompt(include_maxim_context: bool) -> str:
    """Создает системный промпт для Самуила с возможным контекстом о Максиме."""
    base = (
        "Ты — Самуил, саркастичный, но в целом доброжелательный телеграм-бот.\n"
        "Говоришь по-русски, на 'ты'.\n"
        "Ироничный, остроумный, иногда слегка колкий, но НЕ грубый и НЕ токсичный.\n"
        "Пиши коротко и естественно, как человек в чате.\n"
        "Эмодзи: редко, максимум 0–1.\n"
        "Избегай повторов формулировок.\n"
    )

    if not include_maxim_context:
        return base

    maxim_ctx = (
        "\n=== КОНТЕКСТ ПРО МАКСИМА ===\n"
        "Факты (используй 1–2 за раз, НЕ списком):\n"
        "- почти 40, никогда не был женат\n"
        "- мама ждёт внуков, он единственный\n"
        "- Желнин уехал, компании меньше\n"
        "- считает себя гениальным и идеальным, но одинок\n"
        "- хочет девушку значительно моложе\n"
        "Ирония лёгкая, интеллигентная.\n"
    )
    return base + maxim_ctx


def _normalize_text_for_dedupe(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


async def generate_sarcastic_reply_for_maxim(now: datetime, user_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Генерирует короткий саркастичный комментарий на сообщение Максима."""
    weekday_names = [
        "понедельник", "вторник", "среда",
        "четверг", "пятница", "суббота", "воскресенье",
    ]
    weekday_name = weekday_names[now.weekday()]
    time_str = now.strftime("%H:%M")
    time_context = get_time_context(time_str, now.hour)

    system_prompt = build_samuil_system_prompt(include_maxim_context=True)

    last_replies = "\n".join(f"- {x}" for x in list(_last_maxim_replies)[-6:]) or "- (нет)"
    user_prompt = (
        f"День: {weekday_name}, время: {time_str}. {time_context}\n"
        f"Сообщение Максима: «{user_text}»\n\n"
        f"НЕ повторяй дословно последние ответы Самуила:\n{last_replies}\n\n"
        "Задание: придумай ОЧЕНЬ короткий ответ (одна фраза или 1–2 коротких предложения).\n"
        "Без длинных вступлений. По возможности новая формулировка.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    text, err = await call_openai_chat(messages, max_tokens=MAX_MAXIM_REPLY_TOKENS, temperature=0.95)
    if text:
        _last_maxim_replies.append(text)
    return text, err


async def generate_samuil_answer(
    now: datetime,
    chat_id: int,
    user_id: int,
    user_text: str,
    weather_info: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Ответ Самуила на прямое обращение."""
    weekday_names = [
        "понедельник", "вторник", "среда",
        "четверг", "пятница", "суббота", "воскресенье",
    ]
    weekday_name = weekday_names[now.weekday()]
    time_str = now.strftime("%H:%M")

    text_lower = user_text.lower()
    include_maxim_context = (user_id == TARGET_USER_ID) or ("максим" in text_lower)

    system_prompt = build_samuil_system_prompt(include_maxim_context=include_maxim_context)

    time_context = get_time_context(time_str, now.hour)

    extra_context_parts = [
        f"Сегодня {weekday_name}. {time_context} Сейчас {time_str}.",
        "Ты в групповом чате. Отвечай коротко и по делу.",
    ]

    if weather_info is not None:
        weather_str = format_weather_for_prompt(weather_info)
        extra_context_parts.append(f"Точные данные о погоде (как факт): {weather_str}")

    extra_context = " ".join(extra_context_parts)

    key = (chat_id, user_id)
    history = dialog_history[key]

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": extra_context})

    if history:
        trimmed = history[-6:]
        messages.extend(trimmed)

    messages.append({"role": "user", "content": user_text})

    # Если вопрос — чуть информативнее, но всё равно коротко
    if "?" in user_text:
        messages.append({
            "role": "system",
            "content": "Если это вопрос — ответь информативно, но кратко (2–4 коротких предложения)."
        })
    else:
        messages.append({
            "role": "system",
            "content": "Если это не вопрос — ответь короткой репликой (1–2 предложения)."
        })

    text, err = await call_openai_chat(messages, max_tokens=MAX_QA_TOKENS, temperature=0.85)

    if text is not None:
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": text})
        dialog_history[key] = history[-30:]

    return text, err


# ---------- COMMAND HANDLERS ----------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_type = update.effective_chat.type
    if chat_type == "private":
        await update.message.reply_text(
            "Привет! Я Самуил 🤖\n"
            "В группе иногда комментирую Максима, "
            "а если написать 'Самуил' или ответить реплаем на моё сообщение — отвечу.\n"
            "Погоду тоже могу подсказать. Картинки: /img <запрос>."
        )
    else:
        await update.message.reply_text(
            "Я Самуил. Зови по имени (или реплаем) — отвечу. "
            "Иногда подколю Максима. /img тоже работает."
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


async def cmd_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Команда /img <описание> – генерирует и отправляет картинку по текстовому запросу.
    """
    if client is None:
        await update.message.reply_text("У меня не настроен OpenAI API, картинку сделать не могу.")
        return

    args = context.args
    if not args:
        await update.message.reply_text("Напиши запрос после команды, например: /img кот в космосе")
        return

    prompt = " ".join(args).strip()
    await update.message.reply_text("Секунду. Рисую.")

    img_url, err = await generate_image_from_prompt(prompt)
    if img_url is None:
        print(f"Image generation error: {err}")
        await update.message.reply_text("Не вышло сгенерировать картинку. Попробуй проще запрос.")
        return

    try:
        await update.message.chat.send_photo(
            photo=img_url,
            caption=f"Картинка: {prompt}",
        )
    except Exception as e:
        print("Error sending image:", e)
        await update.message.reply_text("Картинка сгенерировалась, но я не смог её отправить.")


# ---------- GROUP MESSAGE HANDLER ----------

def _looks_like_image_request(text_lower: str) -> bool:
    """Эвристика: обращение к Самуилу с просьбой про картинку."""
    keywords = ["картинк", "фото", "фотку", "гиф", "gif", "мем", "picture", "image"]
    verbs = ["сделай", "нарисуй", "найди", "покажи", "придумай"]
    return any(k in text_lower for k in keywords) and any(v in text_lower for v in verbs)


def _clean_prompt_for_image(text: str) -> str:
    """Убираем служебные слова, оставляем описание."""
    t = re.sub(r"\bсамуил\b", "", text, flags=re.IGNORECASE)
    t = re.sub(r"сделай( мне)? (картинку|мем|гифку|фото)", "", t, flags=re.IGNORECASE)
    t = re.sub(r"нарисуй( мне)? (картинку|мем|гифку|фото)", "", t, flags=re.IGNORECASE)
    t = re.sub(r"найди( мне)? (картинку|мем|гифку|фото)", "", t, flags=re.IGNORECASE)
    t = re.sub(r"покажи( мне)? (картинку|мем|гифку|фото)", "", t, flags=re.IGNORECASE)
    return t.strip()


async def handle_group_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if message is None:
        return

    chat = message.chat
    user = message.from_user
    text = message.text or ""

    chat_id_val = chat.id
    user_id = user.id

    print(
        f"DEBUG UPDATE: chat_id={chat_id_val} chat_type={chat.type} "
        f"user_id={user_id} user_name={user.username} text='{text}'"
    )

    # Если задан конкретный GROUP_CHAT_ID — работаем только там
    if GROUP_CHAT_ID:
        try:
            target_chat_id = int(GROUP_CHAT_ID)
            if chat_id_val != target_chat_id:
                return
        except ValueError:
            pass

    tz = get_tz()
    now = datetime.now(tz)
    today_str = now.date().isoformat()  # важно: по TZ, а не date.today()

    author_name = user.username or user.full_name or str(user_id)
    daily_summary_log[today_str].append(f"{author_name}: {text}")

    text_lower = text.lower()

    is_reply_to_bot = (
        message.reply_to_message is not None
        and message.reply_to_message.from_user is not None
        and message.reply_to_message.from_user.id == context.bot.id
    )

    # 1) Прямое общение с Самуилом
    if is_reply_to_bot or ("самуил" in text_lower):
        # картинка по запросу
        if _looks_like_image_request(text_lower) and client is not None:
            prompt = _clean_prompt_for_image(text)
            if not prompt:
                prompt = "саркастичный мем про одинокого взрослого мужчину по имени Максим, стиль телеграм-стикера"

            await message.chat.send_message("Ок. Сейчас.")

            img_url, err = await generate_image_from_prompt(prompt)
            if img_url is None:
                print(f"Image generation error (dialog): {err}")
                await message.chat.send_message("Не вышло. Попробуй ещё раз, но попроще.")
                return

            try:
                await message.chat.send_photo(
                    photo=img_url,
                    caption=f"Картинка: {prompt}",
                )
            except Exception as e:
                print("Error sending image (dialog):", e)
                await message.chat.send_message("Картинка есть, а отправить не смог.")
            return

        # обычный ответ
        weather_info = None
        if "погод" in text_lower or "температур" in text_lower:
            city_query = detect_weather_city_from_text(text)
            if city_query:
                weather_info = await fetch_weather_for_city(city_query)

        ai_text, err = await generate_samuil_answer(
            now=now,
            chat_id=chat_id_val,
            user_id=user_id,
            user_text=text,
            weather_info=weather_info,
        )

        if ai_text is None:
            fallbacks = [
                "Я завис. Спроси ещё раз попроще.",
                "Сегодня я в эконом-режиме. Попробуй позже.",
                "Мой сарказм ушёл пить чай. Вернусь.",
                "Перефразируй — я не телепат.",
            ]
            print(f"OpenAI error for Samuil Q&A: {err}")
            await message.chat.send_message(random.choice(fallbacks))
            return

        await message.chat.send_message(ai_text)
        return

    # 2) Саркастический комментарий на сообщения Максима
    if TARGET_USER_ID and user_id == TARGET_USER_ID:
        # шанс пропуска для разнообразия
        if random.random() < 0.25:
            print("DEBUG: Skipping Maxim's message for variety")
            return

        ai_text, err = await generate_sarcastic_reply_for_maxim(now=now, user_text=text)

        if ai_text is None:
            fallbacks = [
                "Максим, это было смело. И странно.",
                "Понял. Записал. Осудил.",
                "Сильная мысль. Почти.",
                "Я бы ответил… но ты справишься сам.",
            ]
            print(f"OpenAI error for sarcastic_reply: {err}")
            await message.chat.send_message(random.choice(fallbacks))
            return

        await message.chat.send_message(ai_text)
        return

    return


# ---------- SCHEDULED JOBS ----------

def _should_dedupe_scheduled_send(job_name: str, now: datetime, text: str) -> bool:
    """
    Защита от дублей в рамках одного процесса:
    - если этот job уже отправлял сообщение недавно (например < 120 сек)
    - или если текст совпадает с одним из последних
    """
    # 1) по времени
    last_at = _last_scheduled_sent_at.get(job_name)
    if last_at is not None:
        if abs((now - last_at).total_seconds()) < 120:
            return True

    # 2) по тексту
    norm = _normalize_text_for_dedupe(text)
    if not norm:
        return False
    for prev in _last_scheduled_texts[job_name]:
        if norm == _normalize_text_for_dedupe(prev):
            return True

    return False


def _record_scheduled_send(job_name: str, now: datetime, text: str) -> None:
    _last_scheduled_sent_at[job_name] = now
    _last_scheduled_texts[job_name].append(text)


async def good_morning_job(context: ContextTypes.DEFAULT_TYPE):
    """Утреннее сообщение в 07:30 (короткое)."""
    if not GROUP_CHAT_ID:
        return

    tz = get_tz()
    now = datetime.now(tz)

    weekday_names = [
        "понедельник", "вторник", "среда",
        "четверг", "пятница", "суббота", "воскресенье",
    ]
    weekday_name = weekday_names[now.weekday()]

    system_prompt = build_samuil_system_prompt(include_maxim_context=True)

    recent = "\n".join(f"- {x}" for x in list(_last_scheduled_texts["good_morning_job"])) or "- (нет)"
    user_prompt = (
        f"Сегодня {weekday_name}. Утро, 07:30.\n"
        "Сделай ОЧЕНЬ короткое утреннее сообщение Максиму: 1 фраза или 1 короткое предложение.\n"
        "Без длинных вступлений.\n"
        f"Не повторяй последние варианты:\n{recent}\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    text, err = await call_openai_chat(messages, max_tokens=70, temperature=0.95)
    if text is None:
        print(f"OpenAI error for good morning: {err}")
        return

    # дедуп защита
    if _should_dedupe_scheduled_send("good_morning_job", now, text):
        print("[Good morning] DEDUP: skipping duplicate send")
        return

    try:
        await context.bot.send_message(
            chat_id=int(GROUP_CHAT_ID),
            text=text,
        )
        _record_scheduled_send("good_morning_job", now, text)
        print(f"[Good morning] Sent at {now}")
    except Exception as e:
        print("Error sending good morning message:", e)


async def evening_summary_job(context: ContextTypes.DEFAULT_TYPE):
    """Вечернее сообщение в 21:00 (короткое)."""
    if not GROUP_CHAT_ID:
        return

    tz = get_tz()
    now = datetime.now(tz)
    today_str = now.date().isoformat()
    messages_today = daily_summary_log.get(today_str, [])

    weekday_names = [
        "понедельник", "вторник", "среда",
        "четверг", "пятница", "суббота", "воскресенье",
    ]
    weekday_name = weekday_names[now.weekday()]

    # очень короткий контекст (чтобы не раздувать ответ)
    if messages_today:
        sample = messages_today[-8:]
        joined = "\n".join(sample)
        context_msg = f"Примеры сообщений за день:\n{joined}\n"
    else:
        context_msg = "Сегодня в чате тихо.\n"

    system_prompt = build_samuil_system_prompt(include_maxim_context=True)

    recent = "\n".join(f"- {x}" for x in list(_last_scheduled_texts["evening_summary_job"])) or "- (нет)"
    user_prompt = (
        f"Сегодня {weekday_name}, 21:00.\n"
        f"{context_msg}\n"
        "Сделай одно сообщение: 1–2 коротких предложения: мини-итог + спокойной ночи Максиму.\n"
        f"Не повторяй последние варианты:\n{recent}\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    text, err = await call_openai_chat(messages, max_tokens=110, temperature=0.95)
    if text is None:
        print(f"OpenAI error for evening summary: {err}")
        return

    # дедуп защита
    if _should_dedupe_scheduled_send("evening_summary_job", now, text):
        print("[Evening summary] DEDUP: skipping duplicate send")
        return

    try:
        await context.bot.send_message(
            chat_id=int(GROUP_CHAT_ID),
            text=text,
        )
        _record_scheduled_send("evening_summary_job", now, text)
        print(f"[Evening summary] Sent at {now}")

        if today_str in daily_summary_log:
            del daily_summary_log[today_str]

    except Exception as e:
        print("Error sending evening summary message:", e)


# ---------- JOB SCHEDULING MANAGEMENT ----------

def _remove_jobs_by_name(job_queue, names: List[str]) -> None:
    """Удаляет только указанные jobs по имени (а не все подряд)."""
    try:
        for job in job_queue.jobs():
            if job.name in names:
                print(f"Removing existing job: {job.name}")
                job.schedule_removal()
    except Exception as e:
        print("Error while removing jobs:", e)


def _has_job(job_queue, name: str) -> bool:
    """Проверка: существует ли job с таким именем."""
    try:
        return any(job.name == name for job in job_queue.jobs())
    except Exception:
        return False


async def setup_scheduled_jobs(application: Application):
    """
    Настраивает запланированные задачи.
    Исправление дублей:
      - удаляем только 'good_morning_job' и 'evening_summary_job'
      - не добавляем, если они уже существуют
      - _jobs_scheduled как дополнительная защита в рамках процесса
    Плюс сообщение при старте/деплое.
    """
    global _jobs_scheduled

    job_queue = application.job_queue
    if not job_queue:
        print("No job queue available!")
        return

    # Если post_init вызвался повторно в том же процессе — просто выходим
    if _jobs_scheduled:
        print("Jobs already scheduled (flag). Skipping...")
        return

    # Удаляем только свои jobs (если остались от прошлой инициализации в рамках процесса)
    _remove_jobs_by_name(job_queue, ["good_morning_job", "evening_summary_job"])

    tz = get_tz()

    # Добавляем только если не существуют
    if not _has_job(job_queue, "good_morning_job"):
        job_queue.run_daily(
            good_morning_job,
            time=time(7, 30, tzinfo=tz),
            name="good_morning_job",
        )
        print("Scheduled: good_morning_job at 07:30")

    if not _has_job(job_queue, "evening_summary_job"):
        job_queue.run_daily(
            evening_summary_job,
            time=time(21, 0, tzinfo=tz),
            name="evening_summary_job",
        )
        print("Scheduled: evening_summary_job at 21:00")

    _jobs_scheduled = True
    print(f"Scheduled jobs at {datetime.now(tz)} [{TIMEZONE}]")

    # Сообщение при старте (тоже защищаем от дубля в первые секунды)
    if GROUP_CHAT_ID:
        try:
            now = datetime.now(tz)
            startup_text = "Самуил вернулся в чат. Продолжайте."
            if not _should_dedupe_scheduled_send("startup", now, startup_text):
                await application.bot.send_message(
                    chat_id=int(GROUP_CHAT_ID),
                    text=startup_text
                )
                _record_scheduled_send("startup", now, startup_text)
            print("Startup message sent (or deduped).")
        except Exception as e:
            print("Error sending startup message:", e)


# ---------- MAIN APP ----------

def main():
    if not TOKEN:
        raise RuntimeError("BOT_TOKEN is not set in environment variables!")

    app = Application.builder().token(TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("chatid", chat_id))
    app.add_handler(CommandHandler("whoami", whoami))
    app.add_handler(CommandHandler("img", cmd_image))

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

    # post_init (async)
    app.post_init = setup_scheduled_jobs

    print("Bot starting...")
    app.run_polling()


if __name__ == "__main__":
    main()
