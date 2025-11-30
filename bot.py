import os
import random
import asyncio
from datetime import datetime, time, date
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

# ================== НАСТРОЙКИ И ОКРУЖЕНИЕ ==================

TOKEN = os.environ.get("BOT_TOKEN")
GROUP_CHAT_ID = os.environ.get("GROUP_CHAT_ID")  # например "-4046709160"
TIMEZONE = os.environ.get("BOT_TZ", "Australia/Brisbane")

# Telegram user IDs
TARGET_USER_ID = int(os.environ.get("TARGET_USER_ID", "0"))  # Максим
OWNER_CHAT_ID = os.environ.get("OWNER_CHAT_ID")  # Личные тех. сообщения (необязательно)

# OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Погода (опционально; если нет ключа, бот просто не будет использовать погоду)
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")

client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

# ========== ГЛОБАЛЬНЫЕ СТРУКТУРЫ ПАМЯТИ (В РАМКАХ ПРОЦЕССА) ==========

# История диалогов с Самуилом: user_id -> список сообщений для OpenAI
conversation_history: Dict[int, List[Dict[str, str]]] = {}

MAX_DIALOG_HISTORY = 12  # сколько последних сообщений храним на пользователя

# Сообщения за день в целевом чате (для вечернего обзора)
day_messages: List[Tuple[datetime, int, str]] = []
current_day: date = date.today()


# ================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==================

def get_tz() -> pytz.BaseTzInfo:
    return pytz.timezone(TIMEZONE)


def is_night_time(dt: datetime) -> bool:
    """Ночь: с 22:00 включительно до 07:00 (07:00 уже не ночь)."""
    return dt.hour >= 22 or dt.hour < 7


async def log_to_owner(context: ContextTypes.DEFAULT_TYPE, message: str):
    if OWNER_CHAT_ID:
        try:
            await context.bot.send_message(chat_id=int(OWNER_CHAT_ID), text=message)
        except Exception as e:
            print("Failed to send owner log:", e)


async def fetch_weather_brief(city: str) -> Optional[str]:
    """
    Короткая строка с погодой в формате:
    'В Брисбене сейчас 24°C, небольшой дождь.'
    city: 'Brisbane,AU' или 'Kaluga,RU'
    """
    if not OPENWEATHER_API_KEY:
        return None

    try:
        async with httpx.AsyncClient(timeout=10.0) as http_client:
            resp = await http_client.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={
                    "q": city,
                    "appid": OPENWEATHER_API_KEY,
                    "units": "metric",
                    "lang": "ru",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            desc = data["weather"][0]["description"]
            temp = round(data["main"]["temp"])
            if city.startswith("Brisbane"):
                city_name = "Брисбене"
            elif city.startswith("Kaluga"):
                city_name = "Калуге"
            else:
                city_name = city
            return f"В {city_name} сейчас {temp}°C, {desc}."
    except Exception as e:
        print("Weather error:", e)
        return None


async def call_openai_messages(
    messages: List[Dict[str, str]],
    max_tokens: int = 120,
    temperature: float = 0.7,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Универсальная обёртка над OpenAI.
    Принимает готовый список сообщений (system+user+history),
    возвращает (text, error_message).
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
        return text, None
    except Exception as e:
        err = f"Error calling OpenAI: {e}"
        print(err)
        return None, err


async def call_openai_simple(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 120,
    temperature: float = 0.7,
) -> Tuple[Optional[str], Optional[str]]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return await call_openai_messages(messages, max_tokens=max_tokens, temperature=temperature)


def trim_history(history: List[Dict[str, str]], limit: int) -> List[Dict[str, str]]:
    if len(history) <= limit:
        return history
    return history[-limit:]


# ================== ГЕНЕРАЦИЯ ТЕКСТОВ ДЛЯ РАЗНЫХ СЛУЧАЕВ ==================

async def generate_sarcastic_reply_for_maxim(now: datetime, maxim_text: str) -> str:
    """
    Саркастичный ответ Максиму (любой его текст в чате).
    Учитывает доп. контекст про Максима.
    """
    weekday = now.weekday()
    weekday_names = [
        "понедельник", "вторник", "среда", "четверг",
        "пятница", "суббота", "воскресенье",
    ]
    weekday_name = weekday_names[weekday]
    time_str = now.strftime("%H:%M")

    system_prompt = (
        "Ты бот-друг по имени Самуил. Твой стиль — доброжелательный, но саркастичный, "
        "иногда с чёрным юмором, но без настоящей жестокости и оскорблений.\n"
        "Ты общаешься по-русски и на 'ты'.\n\n"
        "Контекст про Максима:\n"
        "- Ему почти 40, он никогда не был женат и живёт один.\n"
        "- Его мама давно ждёт внуков, а Максим у неё единственный ребёнок.\n"
        "- Лучший друг Желнин уехал из Австралии и фактически кинул Максима, "
        "поэтому пить по выходным и петь под гитару Максиму особенно не с кем.\n"
        "- Максим считает себя идеальным и гениальным.\n"
        "- С женщинами у него не складывается: он мечтает о молодой 'лесной нимфе', "
        "но взаимности там не очень.\n\n"
        "Твоя задача — коротко (1–2 предложения) подколоть Максима, используя этот контекст, "
        "но не переходя в прямые оскорбления. Эмодзи можно, но не обязательно, 0–2 штуки."
    )

    user_prompt = (
        f"Сегодня {weekday_name}, время {time_str}. "
        f"Максим написал в чат: «{maxim_text}».\n"
        "Ответь как Самуил. Сделай саркастичный комментарий, который можно прочитать "
        "самостоятельно, без цитирования сообщения Максима."
    )

    text, err = await call_openai_simple(
        system_prompt,
        user_prompt,
        max_tokens=80,
        temperature=0.9,
    )

    if text is None:
        print("OpenAI error in sarcastic reply:", err)
        return "Максим, я даже не знаю, что сказать… ты сам понял, что написал? 🤦‍♂️"

    return text


async def generate_weekday_morning_message(now: datetime) -> str:
    """
    Утреннее сообщение по будням в 7:00, с погодой в Брисбене (если доступна).
    """
    weekday = now.weekday()
    weekday_names = [
        "понедельник", "вторник", "среда", "четверг",
        "пятница", "суббота", "воскресенье",
    ]
    weekday_name = weekday_names[weekday]
    time_str = now.strftime("%H:%M")

    weather_text = await fetch_weather_brief("Brisbane,AU")
    if weather_text:
        weather_part = f"\nКстати, {weather_text}"
    else:
        weather_part = ""

    system_prompt = (
        "Ты Самуил — ироничный, но тёплый бот-друг Максима в рабочем чате.\n"
        "По будням в 7 утра ты желаешь Максиму доброго утра и хорошего рабочего дня.\n"
        "Пиши по-русски, на 'ты', 1–3 коротких предложения. Можно лёгкий юмор.\n"
        "Не забывай, что Максим любит считать себя гением и идеальным, "
        "но в жизни это не всегда подтверждается — можно деликатно намекать на это."
    )

    user_prompt = (
        f"Сегодня {weekday_name}, время {time_str}. "
        f"Нужно утреннее сообщение Максиму, чтобы он проснулся, пошёл работать "
        f"и слегка улыбнулся.{weather_part}"
    )

    text, err = await call_openai_simple(
        system_prompt,
        user_prompt,
        max_tokens=120,
        temperature=0.8,
    )

    if text is None:
        print("OpenAI error in weekday_morning:", err)
        fallback = "Доброе утро, Максим! Рабочий день сам себя не отработает, так что вперёд — удивляй мир. ☕️"
        if weather_text:
            fallback += f"\n{weather_text}"
        return fallback

    # Добавим погоду, если её нет в ответе и она есть
    if weather_text and weather_text.split(" сейчас ")[0] not in text:
        text += "\n" + weather_text

    return text


async def generate_weekend_regular_message(now: datetime) -> str:
    """
    Сообщение по выходным (примерно раз в 3 часа), с вопросом Максиму как дела.
    С погоды берём только короткое упоминание.
    """
    weekday = now.weekday()
    weekday_names = [
        "понедельник", "вторник", "среда", "четверг",
        "пятница", "суббота", "воскресенье",
    ]
    weekday_name = weekday_names[weekday]
    time_str = now.strftime("%H:%M")

    weather_text = await fetch_weather_brief("Brisbane,AU")
    if weather_text:
        # чуть покороче
        weather_short = weather_text.replace("сейчас", "сегодня")  # просто косметика
    else:
        weather_short = ""

    system_prompt = (
        "Ты Самуил — ироничный, но заботливый бот-друг Максима.\n"
        "По выходным ты иногда напоминаешь о себе и спрашиваешь, как он там живёт.\n"
        "Пиши по-русски, на 'ты', 1–3 предложения. Можно шутки и добрый стёб.\n"
        "Иногда упоминай погоду в Брисбене, если она тебе известна, но без длинных сводок."
    )

    user_prompt = (
        f"Сейчас {weekday_name}, {time_str}. "
        f"Нужно очередное виходное сообщение Максиму: спросить, чем он занят, "
        f"подколоть его одиночество или поиски 'лесной нимфы', но без жестокости.\n"
        f"Короткая информация о погоде (можно использовать, а можно игнорировать): {weather_short}"
    )

    text, err = await call_openai_simple(
        system_prompt,
        user_prompt,
        max_tokens=120,
        temperature=0.9,
    )

    if text is None:
        print("OpenAI error in weekend_regular:", err)
        fallback = "Максим, как твои выходные? Надеюсь, ты развлекаешься не только с ноутбуком. 😏"
        if weather_short:
            fallback += f"\n{weather_short}"
        return fallback

    return text


async def generate_goodnight_message(now: datetime) -> str:
    """
    Сообщение в 21:00 — пожелание спокойной ночи от Самуила, с лёгким юмором.
    """
    weekday = now.weekday()
    weekday_names = [
        "понедельник", "вторник", "среда", "четверг",
        "пятница", "суббота", "воскресенье",
    ]
    weekday_name = weekday_names[weekday]

    system_prompt = (
        "Ты Самуил — саркастичный, но заботливый бот-друг Максима.\n"
        "Сейчас вечер, ты пишешь пожелание спокойной ночи.\n"
        "Пиши по-русски, на 'ты', 1–3 предложения. Можно слегка подшутить над "
        "его одиночеством, мамой, ждущей внуков, или поиском 'лесной нимфы', "
        "но с тёплым оттенком, чтобы Максим не обижался."
    )

    user_prompt = (
        f"Сегодня {weekday_name}, вечер около 21:00. "
        f"Сформулируй пожелание спокойной ночи Максиму от Самуила."
    )

    text, err = await call_openai_simple(
        system_prompt,
        user_prompt,
        max_tokens=100,
        temperature=0.9,
    )

    if text is None:
        print("OpenAI error in goodnight:", err)
        return "Спокойной ночи, Максим. Сны тебе пусть будут поинтереснее, чем твой текущий лайфстайл. 🌙"

    return text


async def generate_daily_summary(now: datetime) -> str:
    """
    Саркастический обзор дня в чате, основываясь на day_messages.
    Если сообщений мало, можно об этом и сказать.
    """
    global day_messages, current_day

    if not day_messages:
        return "Сегодня в чате была такая тишина, что даже я заскучал. Завтра попробуем ещё раз. 😴"

    # Базовый текст для OpenAI
    lines = []
    for dt, uid, text in day_messages:
        ts = dt.strftime("%H:%M")
        label = "Максим" if uid == TARGET_USER_ID else f"user_{uid}"
        lines.append(f"[{ts}] {label}: {text}")

    history_text = "\n".join(lines)

    system_prompt = (
        "Ты Самуил — саркастичный, но не злой бот-друг в небольшом чате.\n"
        "Твоя задача — сделать короткий (3–6 предложений) обзор переписки за день, "
        "с акцентом на Максима: его сообщения, поведение, шутки, жалобы и т.п.\n"
        "Пиши по-русски, на 'ты' (обращаясь к Максиму и остальным участникам). "
        "Разрешён добрый стёб, но без прямых оскорблений и травли."
    )

    user_prompt = (
        "Вот хронология сегодняшних сообщений в чате:\n\n"
        f"{history_text}\n\n"
        "Сделай смешной, но относительно доброжелательный обзор дня. "
        "Можно слегка потроллить Максима за одиночество, маму и поиски 'лесной нимфы'."
    )

    text, err = await call_openai_simple(
        system_prompt,
        user_prompt,
        max_tokens=220,
        temperature=0.9,
    )

    # После генерации очищаем дневной лог
    day_messages = []
    current_day = now.date()

    if text is None:
        print("OpenAI error in daily_summary:", err)
        return "Сегодняшний день в чате лучше промолчать… но завтра у нас будет новый шанс. 😉"

    return text


async def generate_weather_comparison(now: datetime) -> str:
    """
    Сравнение погоды Брисбен / Калуга (раз в день).
    Если API недоступна — мягкий фоллбэк.
    """
    bris = await fetch_weather_brief("Brisbane,AU")
    kal = await fetch_weather_brief("Kaluga,RU")

    if not bris and not kal:
        return "Сегодня без метеоаналитики: погода ушла в оффлайн вместе с API."

    system_prompt = (
        "Ты Самуил — ироничный бот. Сравниваешь погоду в Брисбене и в Калуге, "
        "делая короткий (2–4 предложения) шуточный комментарий. Пиши по-русски."
    )

    user_prompt = (
        f"Информация о погоде:\n"
        f"Брисбен: {bris}\n"
        f"Калуга: {kal}\n\n"
        "Сделай смешное сравнение, можно намекнуть, где Максиму лучше страдать "
        "от одиночества и поиска 'лесной нимфы'."
    )

    text, err = await call_openai_simple(
        system_prompt,
        user_prompt,
        max_tokens=160,
        temperature=0.9,
    )

    if text is None:
        print("OpenAI error in weather_comparison:", err)
        # простой фоллбэк
        parts = []
        if bris:
            parts.append(bris)
        if kal:
            parts.append(kal)
        base = "\n".join(parts)
        if not base:
            base = "С погодой всё сложно, как у Максима с личной жизнью."
        return base

    return text


async def generate_samuil_answer(
    user_id: int,
    now: datetime,
    user_text: str,
) -> str:
    """
    Ответ Самуила на прямое обращение (со словом 'самуил').
    Учитываем историю диалога с этим пользователем.
    Максимский контекст добавляем только если в вопросе есть 'максим'.
    """
    weekday = now.weekday()
    weekday_names = [
        "понедельник", "вторник", "среда", "четверг",
        "пятница", "суббота", "воскресенье",
    ]
    weekday_name = weekday_names[weekday]
    time_str = now.strftime("%H:%M")

    base_system = (
        "Ты Самуил — телеграм-бот с характером: остроумный, ироничный, иногда саркастичный, "
        "но в целом доброжелательный. Ты отвечаешь на вопросы участников чата.\n"
        "Пиши по-русски, стиль живой, разговорный, без канцелярщины. "
        "Допускается лёгкий мат в очень мягкой форме, но лучше обходиться без него."
    )

    context_about_maxim = (
        "\n\nДополнительный контекст, который можно использовать, "
        "ТОЛЬКО если вопрос касается Максима:\n"
        "- Максиму почти 40, он никогда не был женат.\n"
        "- Мама ждёт внуков, а он один у неё.\n"
        "- Лучший друг Желнин уехал из Австралии и оставил Максима пить чай с ботом.\n"
        "- Максим считает себя гением и идеальным.\n"
        "- В любви он ищет молодую девушку лет двадцати, но они его почему-то не выбирают.\n"
        "Если вопрос никак не связан с Максимом, эти детали игнорируй."
    )

    # Добавляем контекст про Максима только если в тексте есть 'максим'
    text_lower = user_text.lower()
    if "максим" in text_lower:
        system_prompt = base_system + context_about_maxim
    else:
        system_prompt = base_system

    # История диалога с этим пользователем
    history = conversation_history.get(user_id, [])
    history = trim_history(history, MAX_DIALOG_HISTORY)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": (
                f"Сегодня {weekday_name}, время {time_str}. "
                f"Пользователь написал: «{user_text}». "
                f"Ответь как Самуил."
            ),
        }
    )

    text, err = await call_openai_messages(
        messages,
        max_tokens=300,  # ограничение длины ответа
        temperature=0.8,
    )

    if text is None:
        print("OpenAI error in samuil_answer:", err)
        return "Сегодня Самуил слегка перегрелся и мысль не оформилось. Попробуй спросить ещё раз попроще."

    # Обновляем историю
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": text})
    conversation_history[user_id] = trim_history(history, MAX_DIALOG_HISTORY)

    return text


# ================== ОБРАБОТЧИКИ КОМАНД ==================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_type = update.effective_chat.type
    if chat_type == "private":
        await update.message.reply_text(
            "Привет! Я Самуил 🤖\n"
            "В группе я:\n"
            "• Саркастично комментирую сообщения Максима.\n"
            "• Отвечаю на вопросы, если ты пишешь мне по имени: *Самуил*.\n"
            "• По будням в 7:00 желаю Максиму доброго утра (с погодой).\n"
            "• По выходным иногда интересуюсь его жизнью.\n"
            "• В 20:30 делаю краткий саркастический обзор дня.\n"
            "• В 21:00 желаю спокойной ночи.\n"
            "Ночью с 22:00 до 7:00 я молчу 😴",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text(
            "Самуил подключился к чату.\n"
            "• Сарказм для Максима включен.\n"
            "• На вопросы отвечаю только, если в сообщении есть моё имя — *Самуил*.\n"
            "• Ночью с 22:00 до 7:00 не беспокою.",
            parse_mode="Markdown",
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
    """Простое эхо в личке, чтобы проверять работоспособность."""
    if update.effective_chat.type != "private":
        return
    text = update.message.text
    await update.message.reply_text(f"Ты написал: {text}")


# ================== ОБРАБОТКА СООБЩЕНИЙ В ГРУППЕ ==================

async def handle_group_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global day_messages, current_day

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

    # Только целевой групповой чат
    if GROUP_CHAT_ID and int(GROUP_CHAT_ID) != chat_id:
        return

    tz = get_tz()
    now = datetime.now(tz)

    # Накопление сообщений для дневного обзора (если день сменился, сбрасываем)
    if now.date() != current_day:
        day_messages = []
        current_day = now.date()
    day_messages.append((now, user_id, text))

    # Ночной режим — не отвечаем (но продолжаем логировать день)
    if is_night_time(now):
        return

    text_lower = text.lower()

    # 1) Сообщения Максима — всегда саркастичный ответ,
    #    даже если нет слова "самуил"
    if TARGET_USER_ID and user_id == TARGET_USER_ID:
        reply = await generate_sarcastic_reply_for_maxim(now, text)
        await message.chat.send_message(reply)
        return

    # 2) Прямое обращение к Самуилу — если в тексте есть слово "самуил"
    if "самуил" in text_lower:
        reply = await generate_samuil_answer(user_id, now, text)
        await message.chat.send_message(reply)
        return

    # 3) Остальные сообщения игнорируем
    return


# ================== ПЛАНИРОВЩИК ЗАДАЧ ==================

async def weekend_random_3h_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Запускается каждую минуту.
    На выходных раз в 3 часа выбирает минуту и шлёт одно сообщение.
    """
    if not GROUP_CHAT_ID:
        return

    tz = get_tz()
    now = datetime.now(tz)
    weekday = now.weekday()  # 0=Mon ... 6=Sun

    # Только суббота и воскресенье
    if weekday < 5:
        return

    # Ночной режим
    if is_night_time(now):
        return

    job = context.job
    if job.data is None:
        job.data = {}

    data = job.data
    current_hour = now.hour
    last_block_hour = data.get("last_block_hour")
    target_minute = data.get("target_minute")
    sent_this_block = data.get("sent_this_block", False)

    # Разбиваем сутки на блоки по 3 часа: 0–2, 3–5, ..., 21–23
    block_start = (current_hour // 3) * 3

    if last_block_hour is None or block_start != last_block_hour:
        target_minute = random.randint(0, 59)
        sent_this_block = False
        data["last_block_hour"] = block_start
        data["target_minute"] = target_minute
        data["sent_this_block"] = sent_this_block
        print(f"[Weekend scheduler] New 3h block starting {block_start}:00, target minute {target_minute}")

    if not sent_this_block and now.minute == target_minute:
        text = await generate_weekend_regular_message(now)
        try:
            await context.bot.send_message(chat_id=int(GROUP_CHAT_ID), text=text)
            data["sent_this_block"] = True
            print(f"[Weekend scheduler] Sent message at {now}")
        except Exception as e:
            print("Error sending weekend 3h message:", e)

    job.data = data


async def weekday_morning_job(context: ContextTypes.DEFAULT_TYPE):
    """Сообщение в 7:00 по будням."""
    if not GROUP_CHAT_ID:
        return

    tz = get_tz()
    now = datetime.now(tz)
    weekday = now.weekday()
    if weekday >= 5:
        return

    text = await generate_weekday_morning_message(now)
    try:
        await context.bot.send_message(chat_id=int(GROUP_CHAT_ID), text=text)
        print(f"[Weekday morning] Sent at {now}")
    except Exception as e:
        print("Error sending weekday morning message:", e)


async def daily_summary_job(context: ContextTypes.DEFAULT_TYPE):
    """Саркастический обзор дня в 20:30."""
    if not GROUP_CHAT_ID:
        return
    tz = get_tz()
    now = datetime.now(tz)
    text = await generate_daily_summary(now)
    try:
        await context.bot.send_message(chat_id=int(GROUP_CHAT_ID), text=text)
        print(f"[Daily summary] Sent at {now}")
    except Exception as e:
        print("Error sending daily summary:", e)


async def goodnight_job(context: ContextTypes.DEFAULT_TYPE):
    """Пожелание спокойной ночи в 21:00."""
    if not GROUP_CHAT_ID:
        return
    tz = get_tz()
    now = datetime.now(tz)
    text = await generate_goodnight_message(now)
    try:
        await context.bot.send_message(chat_id=int(GROUP_CHAT_ID), text=text)
        print(f"[Goodnight] Sent at {now}")
    except Exception as e:
        print("Error sending goodnight message:", e)


async def weather_comparison_job(context: ContextTypes.DEFAULT_TYPE):
    """Сравнение погоды Брисбен / Калуга (например, в 12:00 каждый день)."""
    if not GROUP_CHAT_ID:
        return
    tz = get_tz()
    now = datetime.now(tz)
    text = await generate_weather_comparison(now)
    try:
        await context.bot.send_message(chat_id=int(GROUP_CHAT_ID), text=text)
        print(f"[Weather comparison] Sent at {now}")
    except Exception as e:
        print("Error sending weather comparison:", e)


# ================== ИНИЦИАЛИЗАЦИЯ БОТА ==================

def main():
    if not TOKEN:
        raise RuntimeError("BOT_TOKEN is not set in environment variables!")

    app = Application.builder().token(TOKEN).build()

    # Команды
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("chatid", chat_id))
    app.add_handler(CommandHandler("whoami", whoami))

    # Эхо в личке
    app.add_handler(
        MessageHandler(
            filters.TEXT & filters.ChatType.PRIVATE & ~filters.COMMAND,
            echo_private,
        )
    )

    # Сообщения в группах (для Самуила / Максима)
    app.add_handler(
        MessageHandler(
            filters.TEXT & filters.ChatType.GROUPS & ~filters.COMMAND,
            handle_group_message,
        )
    )

    # Планировщик задач
    job_queue = app.job_queue
    tz = get_tz()
    now = datetime.now(tz)

    print(
        f"Local time now: {now} [{TIMEZONE}]. "
        "Scheduling jobs (weekday morning, weekend 3h messages, summary, goodnight, weather comparison)."
    )

    # Будние утренние сообщения в 7:00 (пн–пт)
    job_queue.run_daily(
        weekday_morning_job,
        time=time(7, 0, tzinfo=tz),
        days=(0, 1, 2, 3, 4),
        name="weekday_morning_job",
    )

    # Выходные: раз в 3 часа (логика внутри, job запускается каждую минуту)
    job_queue.run_repeating(
        weekend_random_3h_job,
        interval=60,  # каждую минуту
        first=0,
        name="weekend_random_3h_job",
        data={},
    )

    # Ежедневный обзор дня в 20:30
    job_queue.run_daily(
        daily_summary_job,
        time=time(20, 30, tzinfo=tz),
        name="daily_summary_job",
    )

    # Спокойной ночи в 21:00
    job_queue.run_daily(
        goodnight_job,
        time=time(21, 0, tzinfo=tz),
        name="goodnight_job",
    )

    # Сравнение погоды в 12:00
    job_queue.run_daily(
        weather_comparison_job,
        time=time(12, 0, tzinfo=tz),
        name="weather_comparison_job",
    )

    print("Bot started and jobs scheduled...")
    app.run_polling()


if __name__ == "__main__":
    main()