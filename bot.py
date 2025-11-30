import os
import asyncio
from datetime import datetime, time as dtime
from typing import Dict, List, Tuple, Optional

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

# Optional: куда слать служебные сообщения (например, тебе в личку)
ADMIN_CHAT_ID = os.environ.get("ADMIN_CHAT_ID")

# OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

# Координаты для погоды
BRISBANE_COORDS = (-27.4679, 153.0278, "Australia/Brisbane")
KALUGA_COORDS = (54.5519, 36.2857, "Europe/Moscow")

# Память диалогов с Самуилом: chat_id -> список сообщений [{"role": "user"/"assistant", "content": "..."}]
CHAT_HISTORY: Dict[int, List[Dict[str, str]]] = {}

# Логи сообщений Максима за день: (chat_id, "YYYY-MM-DD") -> [ "HH:MM: текст", ... ]
DAILY_LOGS: Dict[Tuple[int, str], List[str]] = {}

# Расширенный контекст про Максима
MAXIM_CONTEXT = (
    "Вот что ты знаешь о Максиме, используй это для шуток и саркастичных замечаний, "
    "но не перечисляй эти факты списком и не повторяй их дословно каждый раз:\n"
    "• Максиму почти 40 лет, он до сих пор не женат и никогда не был.\n"
    "• Его мама давно ждёт внуков, а Максим у неё единственный ребёнок.\n"
    "• В Австралию он приехал вместе с другом Желниным, но тот уехал и фактически "
    "«кинул» Максима, оставив его здесь одного без собутыльника и гитарных посиделок.\n"
    "• Максим считает себя идеальным и гениальным.\n"
    "• С выбором женщины у Максима беда: он ищет себе молодую «лесную нимфу», "
    "но обычно он их не интересует.\n"
    "Иногда мягко шути именно об этих вещах, но не скатывайся в злую травлю. "
    "Юмор должен быть дружеским и ироничным, а не жестоким."
)


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


def add_history(chat_id: int, role: str, content: str, max_len: int = 30):
    history = CHAT_HISTORY.get(chat_id, [])
    history.append({"role": role, "content": content})
    if len(history) > max_len:
        history = history[-max_len:]
    CHAT_HISTORY[chat_id] = history


def log_maxim_message(now: datetime, chat_id: int, text: str):
    date_str = now.strftime("%Y-%m-%d")
    key = (chat_id, date_str)
    logs = DAILY_LOGS.get(key, [])
    logs.append(f"{now.strftime('%H:%M')}: {text}")
    DAILY_LOGS[key] = logs


async def call_openai_basic(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 120,
    temperature: float = 0.7,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Обёртка над OpenAI без истории. Возвращает (text, error_message).
    """
    if client is None:
        return None, "OpenAI client is not configured (no API key)."

    try:
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


async def call_openai_with_history(
    chat_id: int,
    system_prompt: str,
    user_content: str,
    max_tokens: int = 600,
    temperature: float = 0.7,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Вызов OpenAI с учётом истории диалога в этом чате.
    История хранится в CHAT_HISTORY[chat_id].
    """
    if client is None:
        return None, "OpenAI client is not configured (no API key)."

    history = CHAT_HISTORY.get(chat_id, [])
    messages = [{"role": "system", "content": system_prompt}] + history + [
        {"role": "user", "content": user_content}
    ]

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
        err = f"Error calling OpenAI with history: {e}"
        print(err)
        return None, err


async def fetch_daily_weather_summary(
    city_label: str,
    lat: float,
    lon: float,
    tz_name: str,
) -> Optional[str]:
    """
    Получить краткую сводку погоды на сегодня для указанного города.
    Использует API Open-Meteo (без ключа).
    Возвращает строку на русском или None, если не удалось.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_probability_max",
            "weathercode",
        ],
        "timezone": tz_name,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client_http:
            r = await client_http.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        daily = data.get("daily", {})

        temps_max = daily.get("temperature_2m_max", [])
        temps_min = daily.get("temperature_2m_min", [])
        prec_probs = daily.get("precipitation_probability_max", [])
        weathercodes = daily.get("weathercode", [])

        if not temps_max or not temps_min or not weathercodes:
            return None

        tmax = temps_max[0]
        tmin = temps_min[0]
        precip = prec_probs[0] if prec_probs else None
        code = weathercodes[0]

        # Простейшее описание по weathercode
        if code == 0:
            desc = "ясно"
        elif code in (1, 2):
            desc = "переменная облачность"
        elif code in (3, 45, 48):
            desc = "облачно или туманно"
        elif code in (51, 53, 55, 56, 57):
            desc = "морось или лёгкий дождь"
        elif code in (61, 63, 65, 80, 81, 82):
            desc = "дождливо"
        elif code in (71, 73, 75, 77, 85, 86):
            desc = "снежно"
        elif code in (95, 96, 99):
            desc = "гроза"
        else:
            desc = "нестабильная погода"

        precip_part = ""
        if precip is not None:
            precip_part = f", шанс осадков около {precip:.0f}%"

        return (
            f"В {city_label} сегодня от {tmin:.0f}° до {tmax:.0f}°, {desc}{precip_part}."
        )
    except Exception as e:
        print(f"Weather fetch error for {city_label}: {e}")
        return None


async def get_weather_context_for_morning() -> str:
    """
    Погода для утреннего сообщения: Брисбен + Калуга.
    """
    b_lat, b_lon, b_tz = BRISBANE_COORDS
    k_lat, k_lon, k_tz = KALUGA_COORDS

    bne = await fetch_daily_weather_summary("Брисбене", b_lat, b_lon, b_tz)
    kal = await fetch_daily_weather_summary("Калуге", k_lat, k_lon, k_tz)

    parts = []
    if bne:
        parts.append(bne)
    if kal:
        parts.append(kal)
    return "\n".join(parts)


async def get_weather_context_for_weekend() -> str:
    """
    Погода для выходных сообщений: только Брисбен.
    """
    b_lat, b_lon, b_tz = BRISBANE_COORDS
    bne = await fetch_daily_weather_summary("Брисбене", b_lat, b_lon, b_tz)
    return bne or ""


# ---------- TEXT GENERATION KINDS ----------

async def generate_message_for_kind(
    kind: str,
    now: datetime,
    user_text: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    kind:
      - "sarcastic_reply"   — ответ Максиму
      - "weekday_morning"   — утреннее сообщение по будням (с погодой)
      - "weekend_regular"   — регулярные сообщения по выходным (с погодой)
      - "daily_summary"     — вечерний анализ за день (20:30)
      - "good_night"        — сообщение в 21:00
    user_text:
      - для sarcastic_reply: текст сообщения Максима
      - для weekday_morning/weekend_regular: строка с погодой
      - для daily_summary: лог сообщений за день
    """
    weekday = now.weekday()  # 0=Mon ... 6=Sun
    weekday_names = [
        "понедельник", "вторник", "среда", "четверг",
        "пятница", "суббота", "воскресенье",
    ]
    weekday_name = weekday_names[weekday]
    time_str = now.strftime("%H:%M")

    if kind == "sarcastic_reply":
        system_prompt = (
            "Ты дружелюбный, но довольно саркастичный бот-друг по имени 'Самуил'. "
            "Ты пишешь по-русски, на 'ты', коротко (1–2 предложения). "
            "Мягко подкалывай Максима, но без реальной жестокости или оскорблений. "
            "Не используй эмодзи в каждом сообщении, максимум один, и не всегда. "
            + MAXIM_CONTEXT
        )
        user_prompt = (
            f"Сегодня {weekday_name}, время {time_str}. "
            f"Максим написал в чат: «{user_text}».\n"
            "Ответь коротко, с лёгкой иронией и юмором. Не повторяй дословно текст Максима. "
            "Сообщение должно быть самостоятельным, а не выглядеть как явный ответ в стиле «ты написал ...»."
        )
        return await call_openai_basic(system_prompt, user_prompt, max_tokens=120, temperature=0.9)

    if kind == "weekday_morning":
        weather_info = user_text or ""
        if weather_info:
            weather_part = (
                "Вот сводка погоды на сегодня для Брисбена и Калуги:\n"
                f"{weather_info}\n"
            )
        else:
            weather_part = (
                "Информации о погоде нет (API недоступно), но сделай вид, что ты всё равно в курсе погоды.\n"
            )

        system_prompt = (
            "Ты бот-друг Самуил в рабочем чате. "
            "По будням в 7 утра ты желаешь Максиму доброго утра и хорошего рабочего дня. "
            "Пиши по-русски, на 'ты', 1–2 предложения. "
            "Лёгкий, доброжелательный тон, можно с лёгким юмором и небольшой иронией. "
            "Обязательно упомяни, что впереди рабочий день. "
            "Сделай короткое сравнение погоды в Брисбене и Калуге. "
            "Эмодзи можно, но не обязательно, не больше одного. "
            + MAXIM_CONTEXT
        )
        user_prompt = (
            f"Сегодня {weekday_name}, время {time_str}.\n"
            f"{weather_part}"
            "Сделай короткое утреннее сообщение для Максима: поздоровайся, "
            "пожелай хорошего рабочего дня, с юмором намекни на его жизнь и привычки, "
            "и аккуратно сравни погоду в Брисбене и Калуге."
        )
        return await call_openai_basic(system_prompt, user_prompt, max_tokens=160, temperature=0.8)

    if kind == "weekend_regular":
        weather_info = user_text or ""
        if weather_info:
            weather_part = (
                "Вот сводка погоды в Брисбене на сегодня:\n"
                f"{weather_info}\n"
            )
        else:
            weather_part = (
                "Информации о погоде нет (API недоступно), но сделай вид, что примерно понимаешь, что там за погода.\n"
            )

        system_prompt = (
            "Ты бот-друг Самуил в чатике. "
            "По выходным несколько раз в день ты пишешь Максиму короткие смешные сообщения с вопросом как дела. "
            "Пиши по-русски, на 'ты', 1–2 предложения. "
            "Тон максимально дружески-саркастичный, но без грубостей. "
            "Можешь слегка шутить про его возраст, одиночество, поиски «лесной нимфы» и прочее из контекста. "
            "Иногда упоминай погоду в Брисбене, но коротко. "
            "Не используй один и тот же смайлик постоянно, если используешь — чередуй."
            + MAXIM_CONTEXT
        )
        user_prompt = (
            f"Сегодня {weekday_name}, время {time_str}.\n"
            f"{weather_part}"
            "Сделай короткое сообщение для Максима: спроси, как он и чем занят, "
            "используя погоду как фон, и добавь немного иронии про его стиль жизни."
        )
        return await call_openai_basic(system_prompt, user_prompt, max_tokens=140, temperature=0.9)

    if kind == "daily_summary":
        logs_text = user_text or ""
        system_prompt = (
            "Ты бот-друг Самуил. Твоя задача — к 20:30 делать саркастическое резюме дня Максима "
            "по его сообщениям в чате. "
            "Пиши по-русски, на 'ты', 3–6 предложений. "
            "Тон — дружеская ирония, можешь шутить довольно жёстко, но не переходи в откровенные оскорбления. "
            "Иногда упоминай его возраст, отсутствие жены, маму, Желнина, поиски «лесной нимфы» и т.д., "
            "но не все сразу и не каждый раз. "
            + MAXIM_CONTEXT
        )
        if logs_text.strip():
            user_prompt = (
                f"Сегодня {weekday_name}, время {time_str}. "
                "Вот сообщения Максима за сегодняшний день (каждое с временем):\n"
                f"{logs_text}\n\n"
                "Сделай ироничный, но тёплый обзор того, каким был его день. "
                "Подчеркни забавные моменты, перекосы, жалобы, попытки выглядеть гениальным и идеальным и т.п."
            )
        else:
            user_prompt = (
                f"Сегодня {weekday_name}, время {time_str}. "
                "Сообщений от Максима за сегодня почти не было.\n"
                "Сделай саркастический комментарий на тему того, что Максим сегодня то ли слишком занят, "
                "то ли снова решил игнорировать чат, и что такое поведение тебя подозревает."
            )

        return await call_openai_basic(system_prompt, user_prompt, max_tokens=200, temperature=0.85)

    if kind == "good_night":
        system_prompt = (
            "Ты бот-друг Самуил. В 21:00 ты желаешь Максиму спокойной ночи. "
            "Пиши по-русски, на 'ты', 1–3 предложения. "
            "Тон — мягко-саркастичный: вроде и доброй ночи желаешь, но и слегка подшучиваешь "
            "над его привычками, одиночеством или планами на сон. "
            "Эмодзи можно, но не обязательно, не больше одного."
            + MAXIM_CONTEXT
        )
        user_prompt = (
            f"Сегодня {weekday_name}, время {time_str}. "
            "Сделай сообщение для Максима с пожеланием спокойной ночи и приятных снов. "
            "Можешь мягко пошутить, что завтра его опять ждёт взрослая жизнь, работа и все его «гениальные» планы."
        )
        return await call_openai_basic(system_prompt, user_prompt, max_tokens=140, temperature=0.8)

    return None, "Unknown message kind"


# ---------- COMMAND HANDLERS ----------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_type = update.effective_chat.type
    if chat_type == "private":
        await update.message.reply_text(
            "Привет! Я Самуил 🤖\n"
            "В группе я буду:\n"
            "• По будням в 7:00 желать Максиму доброго утра и хорошего рабочего дня (с погодой).\n"
            "• По выходным писать ему несколько раз в день с вопросами и шутками, учитывая погоду.\n"
            "• В 20:30 делать саркастический обзор его дня.\n"
            "• В 21:00 желать спокойной ночи.\n"
            "Если в чате написать 'Самуил' и вопрос, я отвечу как маленький ChatGPT."
        )
    else:
        await update.message.reply_text(
            "Я Самуил, местный ИИ-бот.\n"
            "• Будни: сообщение Максиму в 7:00 (с погодой и лёгким сарказмом).\n"
            "• Выходные: несколько сообщений в день для Максима с шутками и погодой.\n"
            "• В 20:30 — обзор его дня.\n"
            "• В 21:00 — пожелание спокойной ночи.\n"
            "Если в сообщении есть моё имя 'Самуил', я отвечаю как умный бот."
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


# ---------- SAMUIL Q&A (по слову "самуил") ----------

async def handle_samuil_question(
    message,
    context: ContextTypes.DEFAULT_TYPE,
    now: datetime,
):
    chat = message.chat
    user = message.from_user
    text = message.text or ""
    chat_id = chat.id

    system_prompt = (
        "Ты ИИ-ассистент по имени Самуил в групповом чате. "
        "Твоя задача — отвечать на вопросы, когда тебя напрямую упоминают по имени "
        "(например, 'Самуил, ...'). "
        "Пиши по-русски, на 'ты', но можешь сохранять лёгкий ироничный тон, "
        "особенно если речь про Максима. "
        "Отвечай содержательно, можно развёрнуто (несколько абзацев), "
        "но без лишней воды и без откровенной грубости. "
        "Если вопрос явно про Максима, используй следующий контекст:\n"
        + MAXIM_CONTEXT
        + "\nЕсли вопрос не про Максима, контекст можно игнорировать и просто отвечать по существу."
    )

    user_content = (
        f"Сейчас {now.strftime('%Y-%m-%d %H:%M')}. "
        f"Сообщение от @{user.username or user.full_name} (id {user.id}) в чате {chat_id}:\n"
        f"{text}\n"
        "Ответь как Самуил, продолжая диалог с учётом предыдущей истории в этом чате."
    )

    ai_text, err = await call_openai_with_history(
        chat_id=chat_id,
        system_prompt=system_prompt,
        user_content=user_content,
        max_tokens=400,
        temperature=0.7,
    )

    if ai_text is None:
        fallback = (
            "Сегодня Самуил немного перегружен нейронами и отвечает коротко: "
            "вопрос я увидел, но пока не готов блеснуть интеллектом. Попробуй ещё раз позже."
        )
        print(f"OpenAI error in Samuil Q&A: {err}")
        await message.chat.send_message(fallback)
        return

    # Обновляем историю
    add_history(chat_id, "user", user_content)
    add_history(chat_id, "assistant", ai_text)

    await message.chat.send_message(ai_text)


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
    lower_text = text.lower()

    # 1) Если упомянули Самуила — это приоритет, отвечаем как ИИ
    if "самуил" in lower_text:
        await handle_samuil_question(message, context, now)
        return

    # 2) Сообщения Максима — автоматический саркастичный ответ + логирование для анализа дня
    if TARGET_USER_ID and user_id == TARGET_USER_ID:
        log_maxim_message(now, chat_id, text)

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

    # 3) Остальные пользователи — бот молчит, если нет слова "Самуил"
    return


# ---------- SCHEDULED JOBS ----------

async def weekday_morning_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Будни, 7:00 — доброе утро Максиму + погода (Брисбен vs Калуга).
    """
    if not GROUP_CHAT_ID:
        return

    tz = get_tz()
    now = datetime.now(tz)
    weekday = now.weekday()
    if weekday >= 5:
        return  # на всякий случай

    weather_context = await get_weather_context_for_morning()

    text, err = await generate_message_for_kind(
        "weekday_morning", now=now, user_text=weather_context
    )
    if text is None:
        text = (
            "Доброе утро, Максим! Погоду я сегодня не нашёл, "
            "но рабочий день всё равно найдёт тебя сам. 😉"
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


async def weekend_regular_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Выходные: несколько сообщений в день с погодой в Брисбене.
    Запускается в заданные часы по расписанию (9:00, 12:00, 15:00, 18:00).
    """
    if not GROUP_CHAT_ID:
        return

    tz = get_tz()
    now = datetime.now(tz)
    weekday = now.weekday()
    if weekday < 5:
        return  # только суббота/воскресенье

    weather_context = await get_weather_context_for_weekend()

    text, err = await generate_message_for_kind(
        "weekend_regular", now=now, user_text=weather_context
    )
    if text is None:
        text = "Максим, как там твои выходные? Погода, конечно, какая-то, но главное — ты. 🤨"
        print(f"OpenAI error for weekend_regular: {err}")

    try:
        await context.bot.send_message(
            chat_id=int(GROUP_CHAT_ID),
            text=text,
        )
        print(f"[Weekend regular] Sent weekend message at {now}")
    except Exception as e:
        print("Error sending weekend regular message:", e)


async def evening_summary_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Каждый день в 20:30 — саркастический анализ сообщений Максима за день.
    """
    if not GROUP_CHAT_ID:
        return

    tz = get_tz()
    now = datetime.now(tz)
    chat_id = int(GROUP_CHAT_ID)
    date_str = now.strftime("%Y-%m-%d")
    key = (chat_id, date_str)

    logs = DAILY_LOGS.pop(key, [])
    logs_text = "\n".join(logs)

    text, err = await generate_message_for_kind(
        "daily_summary", now=now, user_text=logs_text
    )
    if text is None:
        text = (
            "Сегодня Максим был загадочно тих… либо ничего не писал, "
            "либо писал так, что я решил это забыть ради его же репутации."
        )
        print(f"OpenAI error for daily_summary: {err}")

    try:
        await context.bot.send_message(
            chat_id=chat_id,
            text=text,
        )
        print(f"[Evening summary] Sent daily summary at {now}")
    except Exception as e:
        print("Error sending evening summary message:", e)


async def good_night_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Каждый день в 21:00 — пожелание спокойной ночи Максиму.
    """
    if not GROUP_CHAT_ID:
        return

    tz = get_tz()
    now = datetime.now(tz)

    text, err = await generate_message_for_kind(
        "good_night", now=now
    )
    if text is None:
        text = (
            "Спокойной ночи, Максим. Постарайся сегодня хотя бы во сне сделать вид, "
            "что у тебя режим. 😴"
        )
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
        "Scheduling weekday morning, weekend regular messages, evening summary and good night jobs."
    )

    # 1) Будние утренние сообщения в 7:00 (пн–пт)
    job_queue.run_daily(
        weekday_morning_job,
        time=dtime(7, 0, tzinfo=tz),
        days=(0, 1, 2, 3, 4),     # понедельник-пятница
        name="weekday_morning_job",
    )

    # 2) Выходные: 4 раза в день 9:00, 12:00, 15:00, 18:00 (сб, вс)
    for h in (9, 12, 15, 18):
        job_queue.run_daily(
            weekend_regular_job,
            time=dtime(h, 0, tzinfo=tz),
            days=(5, 6),          # суббота, воскресенье
            name=f"weekend_regular_{h}",
        )

    # 3) Ежедневный анализ дня в 20:30
    job_queue.run_daily(
        evening_summary_job,
        time=dtime(20, 30, tzinfo=tz),
        days=(0, 1, 2, 3, 4, 5, 6),
        name="evening_summary_job",
    )

    # 4) Спокойной ночи в 21:00 каждый день
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