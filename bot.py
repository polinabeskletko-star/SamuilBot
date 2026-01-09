import os
import re
import json
import random
import asyncio
import logging
from datetime import datetime, time, date, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any

import pytz
import httpx
from openai import OpenAI, AsyncOpenAI
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
    JobQueue,
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
OPENAI_IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-4o-mini")  # Исправлено

# Используем асинхронного клиента OpenAI
client: Optional[AsyncOpenAI] = None
if OPENAI_API_KEY:
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# OpenWeather
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ---------- GLOBAL STATE ----------

# История диалогов с Самуилом: (chat_id, user_id) -> list[{"role": "...", "content": "..."}]
dialog_history: Dict[Tuple[int, int], List[Dict[str, str]]] = defaultdict(list)

# Логи сообщений для вечернего анализа: date_str -> list[str]
daily_summary_log: Dict[str, List[str]] = defaultdict(list)

# Дедуп отправки плановых сообщений
# job_name -> datetime last_sent_at (tz-aware)
_last_scheduled_sent_at: Dict[str, datetime] = {}
# job_name -> deque последних текстов
_last_scheduled_texts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))

# Для разнообразия ответов Максиму: хранить последние ответы
_last_maxim_replies: deque = deque(maxlen=8)

# Кэш для погоды: city -> (data, timestamp)
_weather_cache: Dict[str, Tuple[Dict[str, Any], datetime]] = {}
WEATHER_CACHE_TTL = 300  # 5 минут

# Кэш для OpenAI ответов: hash -> (response, timestamp)
_openai_cache: Dict[str, Tuple[str, datetime]] = {}
OPENAI_CACHE_TTL = 600  # 10 минут

# ---------- HELPERS ----------

def get_tz() -> pytz.BaseTzInfo:
    return pytz.timezone(TIMEZONE)


def is_night_time(dt: datetime) -> bool:
    """Ночь: с 22:00 включительно до 07:00 (07:00 уже не ночь)."""
    hour = dt.hour
    return hour >= 22 or hour < 7


async def log_to_admin(context: ContextTypes.DEFAULT_TYPE, message: str):
    """Логирование в админский чат."""
    if ADMIN_CHAT_ID:
        try:
            await context.bot.send_message(chat_id=int(ADMIN_CHAT_ID), text=message)
        except Exception as e:
            logger.error(f"Failed to send admin log: {e}")


def generate_cache_key(messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
    """Генерация ключа для кэша OpenAI запросов."""
    import hashlib
    key_str = f"{json.dumps(messages, sort_keys=True)}:{max_tokens}:{temperature}"
    return hashlib.md5(key_str.encode()).hexdigest()


async def call_openai_chat(
    messages: List[Dict[str, str]],
    max_tokens: int = 120,
    temperature: float = 0.7,
    use_cache: bool = True,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Универсальная обёртка над OpenAI chat.completions.
    Использует кэширование для одинаковых запросов.
    """
    if client is None:
        return None, "OpenAI client is not configured (no API key)."

    # Проверяем кэш
    if use_cache:
        cache_key = generate_cache_key(messages, max_tokens, temperature)
        cached_data = _openai_cache.get(cache_key)
        if cached_data:
            response, timestamp = cached_data
            if (datetime.now() - timestamp).total_seconds() < OPENAI_CACHE_TTL:
                logger.debug(f"Using cached OpenAI response for key: {cache_key[:8]}")
                return response, None

    try:
        # Используем асинхронный вызов напрямую
        resp = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return None, "Empty response from OpenAI."
        
        # Сохраняем в кэш
        if use_cache:
            cache_key = generate_cache_key(messages, max_tokens, temperature)
            _openai_cache[cache_key] = (text, datetime.now())
            
        return text, None
    except Exception as e:
        err = f"Error calling OpenAI: {e}"
        logger.error(err)
        return None, err


async def generate_image_from_prompt(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Генерация картинки через OpenAI Images по текстовому запросу.
    Возвращает (image_url, error_message).
    """
    if client is None:
        return None, "OpenAI client is not configured (no API key)."

    try:
        resp = await client.images.generate(
            model="dall-e-3",  # или "dall-e-2" для более дешевого варианта
            prompt=prompt,
            n=1,
            size="1024x1024",
            quality="standard",
        )
        image_url = resp.data[0].url
        return image_url, None
    except Exception as e:
        err = f"Error calling OpenAI Images: {e}"
        logger.error(err)
        return None, err


# ---------- WEATHER HELPERS ----------

async def fetch_weather_for_city(city_query: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
    """
    Получить погоду из OpenWeather по названию города.
    Использует кэширование.
    """
    if not OPENWEATHER_API_KEY:
        logger.warning("No OPENWEATHER_API_KEY configured")
        return None

    # Проверяем кэш
    if use_cache:
        cached_data = _weather_cache.get(city_query)
        if cached_data:
            data, timestamp = cached_data
            if (datetime.now() - timestamp).total_seconds() < WEATHER_CACHE_TTL:
                logger.debug(f"Using cached weather for: {city_query}")
                return data

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
            logger.error(f"OpenWeather error for '{city_query}': {resp.status_code} {resp.text}")
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
        
        # Сохраняем в кэш
        if use_cache:
            _weather_cache[city_query] = (result, datetime.now())
            
        return result
    except Exception as e:
        logger.error(f"Error fetching weather: {e}")
        return None


def detect_weather_city_from_text(text: str) -> Optional[str]:
    """
    Пытаемся понять, для какого города просят погоду.
    """
    t = text.lower()

    # Преобразуем русские названия в английские для API
    city_mapping = {
        "калуге": "Kaluga,ru",
        "калуга": "Kaluga,ru",
        "kaluga": "Kaluga,ru",
        "брисбене": "Brisbane,au",
        "брисбен": "Brisbane,au",
        "brisbane": "Brisbane,au",
        "москве": "Moscow,ru",
        "москва": "Moscow,ru",
        "moscow": "Moscow,ru",
        "питере": "Saint Petersburg,ru",
        "петербург": "Saint Petersburg,ru",
        "спб": "Saint Petersburg,ru",
    }

    for russian, english in city_mapping.items():
        if russian in t:
            return english

    # Попробуем найти упоминание города после "в" или "в городе"
    m = re.search(r"\b(?:в|в городе)\s+([А-Яа-яA-Za-z\-]+)", t)
    if m:
        city_raw = m.group(1)
        # Если город на русском, попробуем его найти в маппинге
        if any(cyr_char in city_raw for cyr_char in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"):
            city_lower = city_raw.lower()
            for russian, english in city_mapping.items():
                if city_lower in russian:
                    return english
        return city_raw

    return None


def format_weather_for_prompt(info: Dict[str, Any]) -> str:
    """Форматирование данных о погоде для промпта."""
    if not info:
        return ""
        
    parts = []
    city = info.get("city")
    country = info.get("country")
    temp = info.get("temp")
    feels = info.get("feels_like")
    hum = info.get("humidity")
    desc = info.get("description")

    if city:
        location = f"{city}, {country}" if country else str(city)
        parts.append(f"Погода в {location}")
    if desc:
        parts.append(f"сейчас {desc}")
    if temp is not None:
        parts.append(f"температура {round(temp)}°C")
    if feels is not None and abs(feels - temp) > 1:
        parts.append(f"ощущается как {round(feels)}°C")
    if hum is not None:
        parts.append(f"влажность {hum}%")

    return ", ".join(parts)


# ---------- AI MESSAGE GENERATORS ----------

MAX_QA_TOKENS = 160
MAX_MAXIM_REPLY_TOKENS = 70
MAX_SCHEDULED_TOKENS = 90

def get_time_context(hour: int) -> str:
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


def build_samuil_system_prompt(include_maxim_context: bool = False) -> str:
    """Создает системный промпт для Самуила."""
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
    """Нормализация текста для дедупликации."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _should_dedupe_scheduled_send(job_name: str, now: datetime, text: str) -> bool:
    """
    Защита от дублей в рамках одного процесса.
    """
    # Нормализуем текст для сравнения
    norm = _normalize_text_for_dedupe(text)
    if not norm:
        return False
    
    # 1) Проверка по времени (не чаще чем раз в 10 минут для scheduled jobs)
    last_at = _last_scheduled_sent_at.get(job_name)
    if last_at is not None:
        time_diff = abs((now - last_at).total_seconds())
        if time_diff < 600:  # 10 минут вместо 5
            logger.info(f"Dedupe: too soon since last send ({time_diff:.0f}s)")
            return True

    # 2) Проверка по тексту (строгая)
    for prev in _last_scheduled_texts[job_name]:
        prev_norm = _normalize_text_for_dedupe(prev)
        if norm == prev_norm:
            logger.info(f"Dedupe: duplicate text detected for {job_name}")
            return True
        
        # Проверка на очень похожие тексты (80% совпадение)
        if len(norm) > 20 and len(prev_norm) > 20:
            # Простая проверка на схожесть
            words_current = set(norm.split())
            words_prev = set(prev_norm.split())
            common_words = words_current.intersection(words_prev)
            similarity = len(common_words) / max(len(words_current), len(words_prev))
            
            if similarity > 0.8:  # 80% совпадение слов
                logger.info(f"Dedupe: high similarity ({similarity:.0%}) for {job_name}")
                return True

    return False


def _record_scheduled_send(job_name: str, now: datetime, text: str) -> None:
    """Запись факта отправки запланированного сообщения."""
    _last_scheduled_sent_at[job_name] = now
    _last_scheduled_texts[job_name].append(text)
    logger.info(f"Recorded send for {job_name} at {now}")


async def generate_sarcastic_reply_for_maxim(now: datetime, user_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Генерация короткого саркастичного комментария на сообщение Максима."""
    weekday_names = [
        "понедельник", "вторник", "среда",
        "четверг", "пятница", "суббота", "воскресенье",
    ]
    weekday_name = weekday_names[now.weekday()]
    time_str = now.strftime("%H:%M")
    time_context = get_time_context(now.hour)

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

    text, err = await call_openai_chat(
        messages, 
        max_tokens=MAX_MAXIM_REPLY_TOKENS, 
        temperature=0.95,
        use_cache=False  # Не кэшируем, так как ответы должны быть уникальными
    )
    
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
    time_context = get_time_context(now.hour)

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
        # Берем последние 4 сообщения для контекста (вместо 6)
        trimmed = history[-4:]
        messages.extend(trimmed)

    messages.append({"role": "user", "content": user_text})

    # Контекст в зависимости от типа сообщения
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

    text, err = await call_openai_chat(
        messages, 
        max_tokens=MAX_QA_TOKENS, 
        temperature=0.85,
        use_cache=False  # Диалоги уникальны
    )

    if text is not None:
        # Ограничиваем историю до 20 сообщений (вместо 30)
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": text})
        if len(history) > 20:
            dialog_history[key] = history[-20:]
        else:
            dialog_history[key] = history

    return text, err


# ---------- COMMAND HANDLERS ----------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start."""
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
    """Показывает ID текущего чата."""
    cid = update.effective_chat.id
    await update.message.reply_text(
        f"Chat ID for this chat: `{cid}`",
        parse_mode="Markdown",
    )


async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает информацию о пользователе."""
    user = update.effective_user
    await update.message.reply_text(
        f"Your user ID: `{user.id}`\nUsername: @{user.username}",
        parse_mode="Markdown",
    )


async def echo_private(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Echo только в личке."""
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
    if len(prompt) > 1000:
        await update.message.reply_text("Запрос слишком длинный. Укороти, пожалуйста.")
        return

    # Показываем статус
    status_msg = await update.message.reply_text("🎨 Создаю картинку...")

    img_url, err = await generate_image_from_prompt(prompt)
    if img_url is None:
        logger.error(f"Image generation error: {err}")
        await status_msg.edit_text("Не вышло сгенерировать картинку. Попробуй проще запрос.")
        return

    try:
        # Удаляем статус-сообщение
        await status_msg.delete()
        
        # Отправляем картинку
        await update.message.chat.send_photo(
            photo=img_url,
            caption=f"🎨 {prompt[:100]}{'...' if len(prompt) > 100 else ''}",
        )
    except Exception as e:
        logger.error(f"Error sending image: {e}")
        await update.message.reply_text("Картинка сгенерировалась, но я не смог её отправить.")


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Очистка истории диалога для пользователя."""
    key = (update.effective_chat.id, update.effective_user.id)
    if key in dialog_history:
        dialog_history[key] = []
        await update.message.reply_text("История диалога очищена.")
    else:
        await update.message.reply_text("У тебя ещё нет истории диалога.")


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает статистику бота."""
    total_dialogs = len(dialog_history)
    total_messages = sum(len(history) for history in dialog_history.values())
    weather_cache_size = len(_weather_cache)
    openai_cache_size = len(_openai_cache)
    
    stats_text = (
        f"📊 Статистика Самуила:\n"
        f"• Активных диалогов: {total_dialogs}\n"
        f"• Всего сообщений в истории: {total_messages}\n"
        f"• Городов в кэше погоды: {weather_cache_size}\n"
        f"• Ответов в кэше OpenAI: {openai_cache_size}\n"
        f"• Последних ответов Максиму: {len(_last_maxim_replies)}"
    )
    
    await update.message.reply_text(stats_text)


# ---------- GROUP MESSAGE HANDLER ----------

def _looks_like_image_request(text_lower: str) -> bool:
    """Эвристика: обращение к Самуилу с просьбой про картинку."""
    keywords = ["картинк", "фото", "фотку", "гиф", "gif", "мем", "picture", "image"]
    verbs = ["сделай", "нарисуй", "найди", "покажи", "придумай"]
    return any(k in text_lower for k in keywords) and any(v in text_lower for v in verbs)


def _clean_prompt_for_image(text: str) -> str:
    """Убираем служебные слова, оставляем описание."""
    patterns = [
        (r"\bсамуил\b", ""),
        (r"(сделай|нарисуй|найди|покажи|придумай)( мне)?\s+(картинку|мем|гифку|фото)", ""),
        (r"пожалуйста\b", ""),
        (r"\s+", " "),
    ]
    
    result = text.strip()
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    return result.strip() or "саркастичный мем про одинокого взрослого мужчину по имени Максим"


async def handle_group_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик сообщений в группе."""
    message = update.message
    if message is None or message.text is None:
        return

    chat = message.chat
    user = message.from_user
    text = message.text.strip()

    chat_id_val = chat.id
    user_id = user.id

    logger.info(f"Group message: chat={chat_id_val} user={user_id} ({user.username}) text='{text[:50]}...'")

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
    today_str = now.date().isoformat()

    # Логируем сообщение для вечернего анализа
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
        # Обработка запроса картинки
        if _looks_like_image_request(text_lower) and client is not None:
            prompt = _clean_prompt_for_image(text)
            
            status_msg = await message.chat.send_message("🎨 Создаю картинку...")
            
            img_url, err = await generate_image_from_prompt(prompt)
            if img_url is None:
                logger.error(f"Image generation error (dialog): {err}")
                await status_msg.edit_text("Не вышло. Попробуй ещё раз, но попроще.")
                return

            try:
                await status_msg.delete()
                await message.chat.send_photo(
                    photo=img_url,
                    caption=f"🎨 {prompt[:100]}{'...' if len(prompt) > 100 else ''}",
                )
            except Exception as e:
                logger.error(f"Error sending image (dialog): {e}")
                await message.chat.send_message("Картинка есть, а отправить не смог.")
            return

        # Обычный ответ Самуила
        weather_info = None
        if any(keyword in text_lower for keyword in ["погод", "температур", "жара", "холод", "дождь"]):
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
            logger.error(f"OpenAI error for Samuil Q&A: {err}")
            await message.chat.send_message(random.choice(fallbacks))
            return

        await message.chat.send_message(ai_text)
        return

    # 2) Саркастический комментарий на сообщения Максима
    if TARGET_USER_ID and user_id == TARGET_USER_ID:
        # Шанс пропуска для разнообразия (увеличили до 40%)
        if random.random() < 0.40:
            logger.debug("Skipping Maxim's message for variety")
            return
            
        # Игнорируем очень короткие сообщения (менее 3 символов)
        if len(text) < 3:
            return

        ai_text, err = await generate_sarcastic_reply_for_maxim(now=now, user_text=text)

        if ai_text is None:
            fallbacks = [
                "Максим, это было смело. И странно.",
                "Понял. Записал. Осудил.",
                "Сильная мысль. Почти.",
                "Я бы ответил… но ты справишься сам.",
            ]
            logger.error(f"OpenAI error for sarcastic_reply: {err}")
            await message.chat.send_message(random.choice(fallbacks))
            return

        await message.chat.send_message(ai_text)
        return


# ---------- SCHEDULED JOBS ----------

async def good_morning_job(context: ContextTypes.DEFAULT_TYPE):
    """Утреннее сообщение в 07:30."""
    if not GROUP_CHAT_ID:
        return

    tz = get_tz()
    now = datetime.now(tz)
    
    # Дополнительная проверка - логгируем вызов
    logger.info(f"[Good morning job] Called at {now}")
    
    # Проверяем, не был ли уже выполнен сегодня
    today_str = now.date().isoformat()
    last_send_key = f"good_morning_sent_{today_str}"
    
    if last_send_key in _last_scheduled_sent_at:
        logger.info(f"[Good morning] Already sent today ({today_str}), skipping")
        return

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

    text, err = await call_openai_chat(
        messages, 
        max_tokens=MAX_SCHEDULED_TOKENS, 
        temperature=0.95,
        use_cache=False
    )
    
    if text is None:
        logger.error(f"OpenAI error for good morning: {err}")
        return

    # Дедупликация
    if _should_dedupe_scheduled_send("good_morning_job", now, text):
        logger.info("[Good morning] DEDUP: skipping duplicate send")
        return

    try:
        await context.bot.send_message(
            chat_id=int(GROUP_CHAT_ID),
            text=text,
        )
        _record_scheduled_send("good_morning_job", now, text)
        # Запоминаем, что отправили сегодня
        _last_scheduled_sent_at[last_send_key] = now
        logger.info(f"[Good morning] Sent at {now}")
    except Exception as e:
        logger.error(f"Error sending good morning message: {e}")


async def evening_summary_job(context: ContextTypes.DEFAULT_TYPE):
    """Вечернее сообщение в 21:00."""
    if not GROUP_CHAT_ID:
        return

    tz = get_tz()
    now = datetime.now(tz)
    
    # Дополнительная проверка - логгируем вызов
    logger.info(f"[Evening summary job] Called at {now}")
    
    # Проверяем, не был ли уже выполнен сегодня
    today_str = now.date().isoformat()
    last_send_key = f"evening_summary_sent_{today_str}"
    
    if last_send_key in _last_scheduled_sent_at:
        logger.info(f"[Evening summary] Already sent today ({today_str}), skipping")
        return
    
    messages_today = daily_summary_log.get(today_str, [])

    weekday_names = [
        "понедельник", "вторник", "среда",
        "четверг", "пятница", "суббота", "воскресенье",
    ]
    weekday_name = weekday_names[now.weekday()]

    # Берем только уникальные сообщения (по пользователям)
    unique_messages = []
    seen_authors = set()
    for msg in reversed(messages_today[-12:]):  # Последние 12 сообщений
        author = msg.split(":", 1)[0] if ":" in msg else "unknown"
        if author not in seen_authors:
            unique_messages.append(msg)
            seen_authors.add(author)
    
    if unique_messages:
        joined = "\n".join(unique_messages[-6:])  # Берем 6 последних уникальных
        context_msg = f"Из сегодняшних сообщений:\n{joined}\n"
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

    text, err = await call_openai_chat(
        messages, 
        max_tokens=MAX_SCHEDULED_TOKENS, 
        temperature=0.95,
        use_cache=False
    )
    
    if text is None:
        logger.error(f"OpenAI error for evening summary: {err}")
        return

    # Дедупликация
    if _should_dedupe_scheduled_send("evening_summary_job", now, text):
        logger.info("[Evening summary] DEDUP: skipping duplicate send")
        return

    try:
        await context.bot.send_message(
            chat_id=int(GROUP_CHAT_ID),
            text=text,
        )
        _record_scheduled_send("evening_summary_job", now, text)
        # Запоминаем, что отправили сегодня
        _last_scheduled_sent_at[last_send_key] = now
        logger.info(f"[Evening summary] Sent at {now}")

        # Очищаем логи за сегодня
        if today_str in daily_summary_log:
            del daily_summary_log[today_str]

    except Exception as e:
        logger.error(f"Error sending evening summary message: {e}")


# ---------- JOB SCHEDULING MANAGEMENT ----------

class JobManager:
    """Менеджер для управления запланированными задачами."""
    
    def __init__(self):
        self.jobs_setup = False
        self.setup_time = None
        self.job_names = set()  # Храним имена созданных задач
        
    async def setup_jobs(self, application: Application):
        """Настройка запланированных задач с защитой от дублей."""
        if self.jobs_setup:
            logger.info("Jobs already set up, skipping...")
            return
            
        job_queue = application.job_queue
        if not job_queue:
            logger.error("No job queue available!")
            return
            
        tz = get_tz()
        now = datetime.now(tz)
        
        # ОЧЕНЬ ВАЖНО: очищаем ВСЕ старые задачи Самуила
        existing_jobs = list(job_queue.jobs())
        jobs_to_remove = []
        
        for job in existing_jobs:
            # Удаляем все задачи с нашими функциями
            if hasattr(job.callback, '__name__'):
                if job.callback.__name__ in ['good_morning_job', 'evening_summary_job']:
                    jobs_to_remove.append(job)
                    
        for job in jobs_to_remove:
            try:
                job.schedule_removal()
                logger.info(f"Removed old job: {job.name}")
            except Exception as e:
                logger.error(f"Error removing job {job.name}: {e}")
        
        # Даем время на удаление
        await asyncio.sleep(1)
        
        # Добавляем новые задачи
        morning_job = job_queue.run_daily(
            good_morning_job,
            time=time(7, 30, tzinfo=tz),
            name=f"samuil_good_morning_{int(now.timestamp())}",
        )
        
        evening_job = job_queue.run_daily(
            evening_summary_job,
            time=time(21, 0, tzinfo=tz),
            name=f"samuil_evening_summary_{int(now.timestamp())}",
        )
        
        if morning_job:
            self.job_names.add(morning_job.name)
        if evening_job:
            self.job_names.add(evening_job.name)
        
        self.jobs_setup = True
        self.setup_time = now
        
        logger.info(f"Jobs scheduled at {now} [{TIMEZONE}]")
        logger.info(f"Morning job: {morning_job.name if morning_job else 'failed'}")
        logger.info(f"Evening job: {evening_job.name if evening_job else 'failed'}")
        
        # Сбрадываем историю дедупликации при старте
        global _last_scheduled_sent_at, _last_scheduled_texts
        _last_scheduled_sent_at.clear()
        _last_scheduled_texts.clear()
        
        # Отправляем сообщение о старте (с задержкой)
        if GROUP_CHAT_ID:
            try:
                # Ждем 5 секунд, чтобы бот точно был готов
                await asyncio.sleep(5)
                
                # Отправляем только если бот недавно запустился
                if datetime.now(tz).timestamp() - now.timestamp() < 30:
                    startup_texts = [
                        "Самуил в сети. Режим наблюдения.",
                        "Система активна. Все датчики в норме.",
                        "Бот запущен. Приступаю к мониторингу.",
                    ]
                    
                    await application.bot.send_message(
                        chat_id=int(GROUP_CHAT_ID),
                        text=random.choice(startup_texts)
                    )
                    logger.info("Startup message sent.")
            except Exception as e:
                logger.error(f"Error sending startup message: {e}")

# Создаем глобальный менеджер задач
job_manager = JobManager()


# ---------- ERROR HANDLING ----------

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Глобальный обработчик ошибок."""
    logger.error(f"Exception while handling an update: {context.error}")
    
    # Логируем в админский чат
    if ADMIN_CHAT_ID:
        try:
            error_msg = f"❌ Ошибка в боте:\n{type(context.error).__name__}: {context.error}"
            await context.bot.send_message(
                chat_id=int(ADMIN_CHAT_ID),
                text=error_msg[:4000]  # Ограничение Telegram
            )
        except Exception as e:
            logger.error(f"Failed to send error to admin: {e}")


# ---------- MAIN APP ----------

def main():
    """Основная функция запуска бота."""
    if not TOKEN:
        raise RuntimeError("BOT_TOKEN is not set in environment variables!")

    # Очищаем глобальное состояние при старте
    global _last_scheduled_sent_at, _last_scheduled_texts
    _last_scheduled_sent_at.clear()
    _last_scheduled_texts.clear()
    
    # Создаем приложение с настройками
    app = Application.builder().token(TOKEN).build()
    
    # Добавляем обработчик ошибок
    app.add_error_handler(error_handler)

    # Команды
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("chatid", chat_id))
    app.add_handler(CommandHandler("whoami", whoami))
    app.add_handler(CommandHandler("img", cmd_image))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("stats", cmd_stats))

    # Echo только в личных чатах
    app.add_handler(
        MessageHandler(
            filters.TEXT & filters.ChatType.PRIVATE & ~filters.COMMAND,
            echo_private,
        )
    )

    # Сообщения в группах
    app.add_handler(
        MessageHandler(
            filters.TEXT & filters.ChatType.GROUPS & ~filters.COMMAND,
            handle_group_message,
        )
    )

    # Настройка задач при старте
    async def post_init(application: Application):
        """Функция, вызываемая после инициализации бота."""
        logger.info("Bot initialized, setting up jobs...")
        await job_manager.setup_jobs(application)
        logger.info("Bot is ready!")

    app.post_init = post_init
    
    # Graceful shutdown
    async def shutdown(application: Application):
        """Функция для корректного завершения работы."""
        logger.info("Shutting down bot...")
        if client:
            await client.close()
        logger.info("Bot shutdown complete.")

    app.post_shutdown = shutdown

    logger.info("Bot starting...")
    
    # Запуск бота
    app.run_polling(
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES,
        close_loop=False,
    )


if __name__ == "__main__":
    main()
