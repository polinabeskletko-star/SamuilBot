import os
import re
import json
import random
import asyncio
import logging
from datetime import datetime, time, date
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any
import uuid

import pytz
import httpx
from openai import AsyncOpenAI
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

# Помогает мгновенно понять, один ли процесс работает
INSTANCE_TAG = os.environ.get("INSTANCE_TAG") or str(uuid.uuid4())[:8]

dialog_history: Dict[Tuple[int, int], List[Dict[str, str]]] = defaultdict(list)
daily_summary_log: Dict[str, List[str]] = defaultdict(list)

# job_name -> datetime last_sent_at (tz-aware)
_last_scheduled_sent_at: Dict[str, datetime] = {}
# job_name -> deque последних текстов
_last_scheduled_texts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))

_last_maxim_replies: deque = deque(maxlen=8)

_weather_cache: Dict[str, Tuple[Dict[str, Any], datetime]] = {}
WEATHER_CACHE_TTL = 300  # 5 минут

_openai_cache: Dict[str, Tuple[str, datetime]] = {}
OPENAI_CACHE_TTL = 600  # 10 минут

# /today output cache (готовый текст)
_onthisday_cache: Dict[str, Tuple[str, datetime]] = {}
ONTHISDAY_CACHE_TTL = 6 * 3600  # 6 часов

# onthisday structured cache (список праздников/событий)
_onthisday_struct_cache: Dict[str, Tuple[Dict[str, Any], datetime]] = {}

# флаги "отправлено сегодня" для scheduled (в рамках процесса)
_sent_day_flags: Dict[str, datetime] = {}

# ---------- HELPERS ----------

def get_tz() -> pytz.BaseTzInfo:
    return pytz.timezone(TIMEZONE)


async def log_to_admin(context: ContextTypes.DEFAULT_TYPE, message: str):
    if ADMIN_CHAT_ID:
        try:
            await context.bot.send_message(chat_id=int(ADMIN_CHAT_ID), text=message)
        except Exception as e:
            logger.error(f"Failed to send admin log: {e}")


def generate_cache_key(messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
    import hashlib
    key_str = f"{json.dumps(messages, sort_keys=True)}:{max_tokens}:{temperature}"
    return hashlib.md5(key_str.encode()).hexdigest()


async def call_openai_chat(
    messages: List[Dict[str, str]],
    max_tokens: int = 120,
    temperature: float = 0.7,
    use_cache: bool = True,
) -> Tuple[Optional[str], Optional[str]]:
    if client is None:
        return None, "OpenAI client is not configured (no API key)."

    if use_cache:
        cache_key = generate_cache_key(messages, max_tokens, temperature)
        cached_data = _openai_cache.get(cache_key)
        if cached_data:
            response, timestamp = cached_data
            if (datetime.now() - timestamp).total_seconds() < OPENAI_CACHE_TTL:
                return response, None

    try:
        resp = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return None, "Empty response from OpenAI."

        if use_cache:
            cache_key = generate_cache_key(messages, max_tokens, temperature)
            _openai_cache[cache_key] = (text, datetime.now())

        return text, None
    except Exception as e:
        err = f"Error calling OpenAI: {e}"
        logger.error(err)
        return None, err


async def generate_image_from_prompt(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    if client is None:
        return None, "OpenAI client is not configured (no API key)."

    try:
        resp = await client.images.generate(
            model="dall-e-3",
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
    if not OPENWEATHER_API_KEY:
        return None

    if use_cache:
        cached_data = _weather_cache.get(city_query)
        if cached_data:
            data, timestamp = cached_data
            if (datetime.now() - timestamp).total_seconds() < WEATHER_CACHE_TTL:
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

        if use_cache:
            _weather_cache[city_query] = (result, datetime.now())

        return result
    except Exception as e:
        logger.error(f"Error fetching weather: {e}")
        return None


def detect_weather_city_from_text(text: str) -> Optional[str]:
    t = text.lower()
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

    m = re.search(r"\b(?:в|в городе)\s+([А-Яа-яA-Za-z\-]+)", t)
    if m:
        return m.group(1)
    return None


def format_weather_for_prompt(info: Dict[str, Any]) -> str:
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
    if feels is not None and temp is not None and abs(feels - temp) > 1:
        parts.append(f"ощущается как {round(feels)}°C")
    if hum is not None:
        parts.append(f"влажность {hum}%")

    return ", ".join(parts)


# ---------- TODAY: HOLIDAYS & EVENTS (Wikimedia On This Day) ----------

def _smart_truncate(text: str, max_len: int = 3900) -> str:
    """
    Умная обрезка под лимит Telegram (4096).
    Стараемся резать по границе пункта/строки/слова, а не посреди.
    """
    if not text or len(text) <= max_len:
        return text

    cut = text[:max_len]

    # 1) По началу следующего буллета
    idx = cut.rfind("\n• ")
    if idx > 0 and idx > max_len * 0.6:
        cut = cut[:idx]
    else:
        # 2) По строке
        idx = cut.rfind("\n")
        if idx > 0 and idx > max_len * 0.6:
            cut = cut[:idx]
        else:
            # 3) По слову
            idx = cut.rfind(" ")
            if idx > 0 and idx > max_len * 0.6:
                cut = cut[:idx]

    return cut.rstrip() + "\n…"


async def fetch_onthisday_struct_ru(d: date, use_cache: bool = True) -> Optional[Dict[str, Any]]:
    """
    Тянем структурированные данные 'в этот день' (ru) и выбираем
    небольшую выборку праздников и событий.
    """
    key = d.isoformat()
    now = datetime.now()

    if use_cache:
        cached = _onthisday_struct_cache.get(key)
        if cached:
            data, ts = cached
            if (now - ts).total_seconds() < ONTHISDAY_CACHE_TTL:
                return data

    mm = f"{d.month:02d}"
    dd = f"{d.day:02d}"
    url = f"https://api.wikimedia.org/feed/v1/wikipedia/ru/onthisday/all/{mm}/{dd}"
    headers = {"User-Agent": f"SamuilBot/1.0 (telegram-bot; onthisday; {INSTANCE_TAG})"}

    try:
        async with httpx.AsyncClient(timeout=12) as http_client:
            resp = await http_client.get(url, headers=headers)

        if resp.status_code != 200:
            logger.error(f"OnThisDay API error: {resp.status_code} {resp.text[:200]}")
            return None

        raw = resp.json()

        def _pick(arr: List[Dict[str, Any]], n: int, require_year: bool = False) -> List[Dict[str, Any]]:
            items = list(arr or [])
            random.shuffle(items)
            out = []
            for it in items:
                if require_year and "year" not in it:
                    continue
                txt = (it.get("text") or "").strip()
                if not txt:
                    continue
                # лёгкая фильтрация очень длинных пунктов
                if len(txt) > 240:
                    continue
                out.append(it)
                if len(out) >= n:
                    break
            return out

        # Для "повода" лучше меньше, но сочнее
        holidays = _pick(raw.get("holidays", []), n=3, require_year=False)
        events = _pick(raw.get("events", []), n=5, require_year=True)

        data_out = {
            "date": f"{dd}.{mm}",
            "holidays": [{"text": (h.get("text") or "").strip()} for h in holidays],
            "events": [{"year": e.get("year"), "text": (e.get("text") or "").strip()} for e in events],
        }

        _onthisday_struct_cache[key] = (data_out, now)
        return data_out

    except Exception as e:
        logger.error(f"Error fetching onthisday struct: {e}")
        return None


async def fetch_onthisday_ru(d: date, use_cache: bool = True, max_len: int = 3900) -> Optional[str]:
    """
    Старый /today: праздники+события списком.
    """
    key = d.isoformat()
    now = datetime.now()

    if use_cache:
        cached = _onthisday_cache.get(key)
        if cached:
            text, ts = cached
            if (now - ts).total_seconds() < ONTHISDAY_CACHE_TTL:
                return text

    data = await fetch_onthisday_struct_ru(d, use_cache=use_cache)
    if not data:
        return None

    ddmm = data["date"]
    holidays = data.get("holidays", [])
    events = data.get("events", [])

    lines: List[str] = []
    title = f"📅 Сегодня ({ddmm})"

    if holidays:
        lines.append("Праздники:")
        for h in holidays:
            lines.append(f"• {h.get('text','').strip()}")

    if events:
        if holidays:
            lines.append("")
        lines.append("События:")
        for e in events[:6]:
            y = e.get("year")
            t = (e.get("text") or "").strip()
            if y and t:
                lines.append(f"• {y}: {t}")
            elif t:
                lines.append(f"• {t}")

    if not holidays and not events:
        lines.append("Сегодня без ярких пунктов по базе. Значит, можно придумать свой повод 🙂")

    text_out = title + "\n" + "\n".join(lines)
    text_out = _smart_truncate(text_out, max_len=max_len)

    _onthisday_cache[key] = (text_out, now)
    return text_out


async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tz = get_tz()
    now = datetime.now(tz)
    text = await fetch_onthisday_ru(now.date())
    if not text:
        await update.message.reply_text("Не смог достать события на сегодня. Попробуй позже.")
        return
    await update.message.reply_text(text)


# ---------- NEW: "ПОВОД ПОДНЯТЬ БОКАЛ" ----------

MAX_TOAST_TOKENS = 220  # чтобы не обрезало посередине

def _format_items_for_prompt(data: Dict[str, Any]) -> str:
    ddmm = data.get("date", "")
    holidays = data.get("holidays", [])
    events = data.get("events", [])

    # Соберём 2-4 пункта всего
    pool: List[str] = []
    for h in holidays:
        t = (h.get("text") or "").strip()
        if t:
            pool.append(f"Праздник: {t}")

    for e in events:
        y = e.get("year")
        t = (e.get("text") or "").strip()
        if t and y:
            pool.append(f"Событие: {y} — {t}")
        elif t:
            pool.append(f"Событие: {t}")

    random.shuffle(pool)
    chosen = pool[:4] if len(pool) >= 4 else pool[:max(2, len(pool))]

    # fallback если пусто
    if not chosen:
        chosen = ["Сегодня база скучает. Придумай повод сам."]

    joined = "\n".join(f"- {x}" for x in chosen)
    return f"Дата: {ddmm}\nФакты дня:\n{joined}"


async def generate_toast_from_onthisday(now: datetime) -> Optional[str]:
    """
    Делает короткий "повод поднять бокал (или чай)" в стиле Самуила.
    Важно: без прямого призыва к злоупотреблению — лёгкая шутка и альтернатива без алкоголя.
    """
    data = await fetch_onthisday_struct_ru(now.date(), use_cache=True)
    if not data:
        return None

    system_prompt = (
        "Ты — Самуил, саркастичный, но доброжелательный телеграм-бот.\n"
        "Говоришь по-русски, на 'ты'.\n"
        "Ироничный, остроумный, НЕ грубый.\n"
        "Эмодзи: максимум 1.\n"
        "Пиши коротко, без длинных вступлений.\n"
        "Тема: «повод поднять бокал» по событиям дня.\n"
        "ВАЖНО: не поощряй ежедневное пьянство. Формулируй как «поднять бокал (или чай/безалк)», добавь мягкое «без фанатизма».\n"
    )

    facts = _format_items_for_prompt(data)

    user_prompt = (
        f"{facts}\n\n"
        "Задание:\n"
        "1) Выбери 2–3 самых забавных/контрастных пункта из фактов.\n"
        "2) Сформулируй «Повод дня» в 4–7 строк, как сообщение в чате.\n"
        "3) Структура:\n"
        "   - Заголовок: «🍷 Повод дня (или чай)»\n"
        "   - 2–3 буллета с фактами, перефразированными смешно и лаконично\n"
        "   - 1 короткая финальная фраза-ирония\n"
        "   - В конце: «без фанатизма» или «можно безалк» (1 раз)\n"
        "4) Не выдумывай факты, опирайся только на данные выше.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    text, err = await call_openai_chat(
        messages, max_tokens=MAX_TOAST_TOKENS, temperature=0.95, use_cache=False
    )
    if not text:
        return None

    return _smart_truncate(text.strip(), max_len=3600)


async def cmd_toast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tz = get_tz()
    now = datetime.now(tz)
    toast = await generate_toast_from_onthisday(now)
    if not toast:
        await update.message.reply_text("Сегодня повод не нашёлся. Значит, ты живёшь правильно.")
        return
    await update.message.reply_text(toast)


# ---------- AI GENERATORS ----------

MAX_QA_TOKENS = 160
MAX_MAXIM_REPLY_TOKENS = 70
MAX_SCHEDULED_TOKENS = 90

def get_time_context(hour: int) -> str:
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


def _should_dedupe_scheduled_send(job_name: str, now: datetime, text: str) -> bool:
    norm = _normalize_text_for_dedupe(text)
    if not norm:
        return False

    last_at = _last_scheduled_sent_at.get(job_name)
    if last_at is not None:
        if abs((now - last_at).total_seconds()) < 600:
            logger.info(f"Dedupe: too soon since last send for {job_name}")
            return True

    for prev in _last_scheduled_texts[job_name]:
        prev_norm = _normalize_text_for_dedupe(prev)
        if norm == prev_norm:
            logger.info(f"Dedupe: duplicate text detected for {job_name}")
            return True

        if len(norm) > 20 and len(prev_norm) > 20:
            words_current = set(norm.split())
            words_prev = set(prev_norm.split())
            similarity = len(words_current & words_prev) / max(len(words_current), len(words_prev))
            if similarity > 0.8:
                logger.info(f"Dedupe: high similarity ({similarity:.0%}) for {job_name}")
                return True

    return False


def _record_scheduled_send(job_name: str, now: datetime, text: str) -> None:
    _last_scheduled_sent_at[job_name] = now
    _last_scheduled_texts[job_name].append(text)


async def generate_sarcastic_reply_for_maxim(now: datetime, user_text: str) -> Tuple[Optional[str], Optional[str]]:
    weekday_names = ["понедельник","вторник","среда","четверг","пятница","суббота","воскресенье"]
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
        "Без длинных вступлений.\n"
    )

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]

    text, err = await call_openai_chat(
        messages, max_tokens=MAX_MAXIM_REPLY_TOKENS, temperature=0.95, use_cache=False
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
    weekday_names = ["понедельник","вторник","среда","четверг","пятница","суббота","воскресенье"]
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
        extra_context_parts.append(f"Точные данные о погоде (как факт): {format_weather_for_prompt(weather_info)}")

    key = (chat_id, user_id)
    history = dialog_history[key]

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": " ".join(extra_context_parts)})

    if history:
        messages.extend(history[-4:])

    messages.append({"role": "user", "content": user_text})

    if "?" in user_text:
        messages.append({"role": "system", "content": "Если это вопрос — ответь кратко (2–4 предложения)."})
    else:
        messages.append({"role": "system", "content": "Если это не вопрос — ответь коротко (1–2 предложения)."})

    text, err = await call_openai_chat(messages, max_tokens=MAX_QA_TOKENS, temperature=0.85, use_cache=False)

    if text is not None:
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": text})
        dialog_history[key] = history[-20:]

    return text, err


# ---------- COMMAND HANDLERS ----------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_type = update.effective_chat.type
    if chat_type == "private":
        await update.message.reply_text(
            "Привет! Я Самуил 🤖\n"
            "В группе иногда комментирую Максима, "
            "а если написать 'Самуил' или ответить реплаем — отвечу.\n"
            "Картинки: /img <запрос>. События дня: /today. Повод дня: /toast."
        )
    else:
        await update.message.reply_text(
            "Я Самуил. Зови по имени (или реплаем) — отвечу. /today — что сегодня за день. /toast — повод дня."
        )


async def chat_id_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    await update.message.reply_text(f"Chat ID for this chat: `{cid}`", parse_mode="Markdown")


async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await update.message.reply_text(
        f"Your user ID: `{user.id}`\nUsername: @{user.username}", parse_mode="Markdown"
    )


async def echo_private(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type != "private":
        return
    text = update.message.text or ""
    await update.message.reply_text(f"Ты написал: {text}")


async def cmd_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

    status_msg = await update.message.reply_text("🎨 Создаю картинку...")
    img_url, err = await generate_image_from_prompt(prompt)
    if img_url is None:
        logger.error(f"Image generation error: {err}")
        await status_msg.edit_text("Не вышло сгенерировать картинку. Попробуй проще запрос.")
        return

    try:
        await status_msg.delete()
        await update.message.chat.send_photo(
            photo=img_url,
            caption=f"🎨 {prompt[:100]}{'...' if len(prompt) > 100 else ''}",
        )
    except Exception as e:
        logger.error(f"Error sending image: {e}")
        await update.message.reply_text("Картинка сгенерировалась, но я не смог её отправить.")


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    key = (update.effective_chat.id, update.effective_user.id)
    dialog_history[key] = []
    await update.message.reply_text("История диалога очищена.")


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📊 Статистика Самуила:\n"
        f"• Активных диалогов: {len(dialog_history)}\n"
        f"• Сообщений в истории: {sum(len(h) for h in dialog_history.values())}\n"
        f"• Кэш погоды: {len(_weather_cache)}\n"
        f"• Кэш OpenAI: {len(_openai_cache)}\n"
        f"• Кэш /today: {len(_onthisday_cache)}\n"
        f"• Кэш /today struct: {len(_onthisday_struct_cache)}\n"
        f"• INSTANCE_TAG: {INSTANCE_TAG}"
    )


# ---------- GROUP MESSAGE HANDLER ----------

def _looks_like_image_request(text_lower: str) -> bool:
    keywords = ["картинк", "фото", "фотку", "гиф", "gif", "мем", "picture", "image"]
    verbs = ["сделай", "нарисуй", "найди", "покажи", "придумай"]
    return any(k in text_lower for k in keywords) and any(v in text_lower for v in verbs)


def _clean_prompt_for_image(text: str) -> str:
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
    message = update.message
    if message is None or message.text is None:
        return

    chat = message.chat
    user = message.from_user
    text = message.text.strip()

    chat_id_val = chat.id
    user_id = user.id

    # Если задан конкретный GROUP_CHAT_ID — работаем только там
    if GROUP_CHAT_ID:
        try:
            if chat_id_val != int(GROUP_CHAT_ID):
                return
        except ValueError:
            pass

    tz = get_tz()
    now = datetime.now(tz)
    today_str = now.date().isoformat()

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
        # Картинка по эвристике
        if _looks_like_image_request(text_lower) and client is not None:
            prompt = _clean_prompt_for_image(text)
            status_msg = await message.chat.send_message("🎨 Создаю картинку...")
            img_url, err = await generate_image_from_prompt(prompt)
            if img_url is None:
                await status_msg.edit_text("Не вышло. Попробуй ещё раз, но попроще.")
                return
            await status_msg.delete()
            await message.chat.send_photo(photo=img_url, caption=f"🎨 {prompt[:100]}")
            return

        # Погода только если явно спрашивают
        weather_info = None
        if any(k in text_lower for k in ["погод", "температур", "жара", "холод", "дождь"]):
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
            await message.chat.send_message("Я завис. Спроси ещё раз попроще.")
            return

        await message.chat.send_message(ai_text)
        return

    # 2) Саркастический комментарий на сообщения Максима
    if TARGET_USER_ID and user_id == TARGET_USER_ID:
        if random.random() < 0.40:
            return
        if len(text) < 3:
            return

        ai_text, err = await generate_sarcastic_reply_for_maxim(now=now, user_text=text)
        if ai_text is None:
            await message.chat.send_message("Понял. Записал. Осудил.")
            return
        await message.chat.send_message(ai_text)
        return


# ---------- SCHEDULED JOBS ----------

async def good_morning_job(context: ContextTypes.DEFAULT_TYPE):
    if not GROUP_CHAT_ID:
        return
    tz = get_tz()
    now = datetime.now(tz)

    today_str = now.date().isoformat()
    flag = f"good_morning_sent_{today_str}"
    if flag in _sent_day_flags:
        return

    system_prompt = build_samuil_system_prompt(include_maxim_context=True)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Сделай ОЧЕНЬ короткое утреннее сообщение Максиму: 1 фраза."}
    ]
    text, err = await call_openai_chat(messages, max_tokens=MAX_SCHEDULED_TOKENS, temperature=0.95, use_cache=False)
    if not text:
        return

    if _should_dedupe_scheduled_send("good_morning_job", now, text):
        return

    await context.bot.send_message(chat_id=int(GROUP_CHAT_ID), text=text)
    _record_scheduled_send("good_morning_job", now, text)
    _sent_day_flags[flag] = now


async def today_toast_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Время 'событий дня', но вместо простого списка — повод поднять бокал (или чай).
    """
    if not GROUP_CHAT_ID:
        return
    tz = get_tz()
    now = datetime.now(tz)

    today_str = now.date().isoformat()
    flag = f"today_toast_sent_{today_str}"
    if flag in _sent_day_flags:
        return

    toast = await generate_toast_from_onthisday(now)
    if not toast:
        # мягкий фолбэк
        mm = f"{now.month:02d}"
        dd = f"{now.day:02d}"
        toast = f"🍷 Повод дня (или чай)\n• Сегодня {dd}.{mm}\n• Повод простой: день всё ещё не развалился.\nФинал: можно безалк."

    if _should_dedupe_scheduled_send("today_toast_job", now, toast):
        return

    await context.bot.send_message(chat_id=int(GROUP_CHAT_ID), text=toast)
    _record_scheduled_send("today_toast_job", now, toast)
    _sent_day_flags[flag] = now


async def evening_summary_job(context: ContextTypes.DEFAULT_TYPE):
    if not GROUP_CHAT_ID:
        return
    tz = get_tz()
    now = datetime.now(tz)

    today_str = now.date().isoformat()
    flag = f"evening_summary_sent_{today_str}"
    if flag in _sent_day_flags:
        return

    messages_today = daily_summary_log.get(today_str, [])

    system_prompt = build_samuil_system_prompt(include_maxim_context=True)
    context_msg = "Сегодня в чате тихо.\n" if not messages_today else "Короткий итог дня."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{context_msg}\nСделай 1–2 предложения: мини-итог + спокойной ночи Максиму."}
    ]
    text, err = await call_openai_chat(messages, max_tokens=MAX_SCHEDULED_TOKENS, temperature=0.95, use_cache=False)
    if not text:
        return

    if _should_dedupe_scheduled_send("evening_summary_job", now, text):
        return

    await context.bot.send_message(chat_id=int(GROUP_CHAT_ID), text=text)
    _record_scheduled_send("evening_summary_job", now, text)
    _sent_day_flags[flag] = now
    daily_summary_log.pop(today_str, None)


# ---------- JOB SCHEDULING MANAGEMENT ----------

class JobManager:
    """
    Менеджер для управления запланированными задачами.
    Lock защищает от двойного вызова setup_jobs в одном процессе.
    """
    JOB_MORNING_NAME = "samuil_good_morning"
    JOB_TODAY_TOAST_NAME = "samuil_today_toast"
    JOB_EVENING_NAME = "samuil_evening_summary"

    def __init__(self):
        self.jobs_setup = False
        self.setup_time = None
        self._lock = asyncio.Lock()
        self._startup_sent = False

    async def _remove_jobs_by_name(self, job_queue, name: str):
        """Удаляем все jobs с конкретным именем (если накопились)."""
        try:
            jobs = job_queue.get_jobs_by_name(name)
        except Exception:
            jobs = [j for j in job_queue.jobs() if getattr(j, "name", None) == name]

        for j in jobs:
            try:
                j.schedule_removal()
                logger.info(f"Removed old job by name: {name}")
            except Exception as e:
                logger.error(f"Error removing job {name}: {e}")

    async def setup_jobs(self, application: Application):
        async with self._lock:
            if self.jobs_setup:
                logger.info("Jobs already set up, skipping...")
                return

            job_queue = application.job_queue
            if not job_queue:
                logger.error("No job queue available!")
                return

            tz = get_tz()
            now = datetime.now(tz)

            # Удаляем старые по фиксированным именам
            await self._remove_jobs_by_name(job_queue, self.JOB_MORNING_NAME)
            await self._remove_jobs_by_name(job_queue, self.JOB_TODAY_TOAST_NAME)
            await self._remove_jobs_by_name(job_queue, self.JOB_EVENING_NAME)

            await asyncio.sleep(0.5)

            # --- ВОТ ГДЕ МЕНЯЕТСЯ ВРЕМЯ ---
            job_queue.run_daily(
                good_morning_job,
                time=time(7, 30, tzinfo=tz),
                name=self.JOB_MORNING_NAME,
            )
            job_queue.run_daily(
                today_toast_job,
                time=time(16, 15, tzinfo=tz),
                name=self.JOB_TODAY_TOAST_NAME,
            )
            job_queue.run_daily(
                evening_summary_job,
                time=time(21, 0, tzinfo=tz),
                name=self.JOB_EVENING_NAME,
            )
            # --------------------------------

            self.jobs_setup = True
            self.setup_time = now

            logger.info(f"Jobs scheduled at {now} [{TIMEZONE}] instance={INSTANCE_TAG}")

            # Сбрасываем дедуп-истории на старте (в рамках одного процесса)
            _last_scheduled_sent_at.clear()
            _last_scheduled_texts.clear()

            # Startup message: защита от двойной отправки в одном процессе
            if GROUP_CHAT_ID and not self._startup_sent:
                try:
                    await asyncio.sleep(2)
                    key = "startup_sent_guard"
                    last = _last_scheduled_sent_at.get(key)
                    if last and abs((datetime.now(tz) - last).total_seconds()) < 60:
                        return

                    startup_texts = [
                        f"Самуил в сети. Режим наблюдения. [{INSTANCE_TAG}]",
                        f"Система активна. Все датчики в норме. [{INSTANCE_TAG}]",
                        f"Бот запущен. Приступаю к мониторингу. [{INSTANCE_TAG}]",
                    ]
                    await application.bot.send_message(
                        chat_id=int(GROUP_CHAT_ID),
                        text=random.choice(startup_texts),
                    )
                    _last_scheduled_sent_at[key] = datetime.now(tz)
                    self._startup_sent = True
                except Exception as e:
                    logger.error(f"Error sending startup message: {e}")


job_manager = JobManager()


# ---------- ERROR HANDLING ----------

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Exception while handling an update: {context.error}")
    if ADMIN_CHAT_ID:
        try:
            error_msg = f"❌ Ошибка в боте [{INSTANCE_TAG}]:\n{type(context.error).__name__}: {context.error}"
            await context.bot.send_message(chat_id=int(ADMIN_CHAT_ID), text=error_msg[:4000])
        except Exception as e:
            logger.error(f"Failed to send error to admin: {e}")


# ---------- MAIN APP ----------

def main():
    if not TOKEN:
        raise RuntimeError("BOT_TOKEN is not set in environment variables!")

    _last_scheduled_sent_at.clear()
    _last_scheduled_texts.clear()
    _sent_day_flags.clear()

    app = Application.builder().token(TOKEN).build()
    app.add_error_handler(error_handler)

    # Команды
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("chatid", chat_id_cmd))
    app.add_handler(CommandHandler("whoami", whoami))
    app.add_handler(CommandHandler("img", cmd_image))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("today", cmd_today))
    app.add_handler(CommandHandler("toast", cmd_toast))

    # Echo только в личке
    app.add_handler(MessageHandler(filters.TEXT & filters.ChatType.PRIVATE & ~filters.COMMAND, echo_private))

    # Сообщения в группах
    app.add_handler(MessageHandler(filters.TEXT & filters.ChatType.GROUPS & ~filters.COMMAND, handle_group_message))

    async def post_init(application: Application):
        logger.info(f"Bot initialized, setting up jobs... instance={INSTANCE_TAG}")
        await job_manager.setup_jobs(application)
        logger.info(f"Bot is ready! instance={INSTANCE_TAG}")

    app.post_init = post_init

    async def shutdown(application: Application):
        logger.info(f"Shutting down bot... instance={INSTANCE_TAG}")
        if client:
            await client.close()
        logger.info("Bot shutdown complete.")

    app.post_shutdown = shutdown

    logger.info(f"Bot starting... instance={INSTANCE_TAG}")

    app.run_polling(
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES,
        close_loop=False,
    )


if __name__ == "__main__":
    main()
