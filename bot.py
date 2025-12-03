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

# ==== НАСТРОЙКИ & ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ ====

TOKEN = os.environ.get("BOT_TOKEN")
GROUP_CHAT_ID = os.environ.get("GROUP_CHAT_ID")  # например, "-1001234567890" (строка)
TIMEZONE = os.environ.get("BOT_TZ", "Australia/Brisbane")

# Telegram user IDs
TARGET_USER_ID = int(os.environ.get("TARGET_USER_ID", "0"))   # Максим
ADMIN_CHAT_ID = os.environ.get("ADMIN_CHAT_ID")

# OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

# Погода (OpenWeather)
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
OPENWEATHER_CITY_ID = os.environ.get("OPENWEATHER_CITY_ID")  # можно использовать ID города или q=...

client = OpenAI(api_key=OPENAI_API_KEY)

# ==== ПОРТРЕТ МАКСИМА ДЛЯ ПРОМТА ====

MAXIM_PROFILE = """
ПОРТРЕТ МАКСИМА (для настройки коммуникации):

Максим — мужчина около сорока лет, живущий в Австралии, но внутренне всё ещё слегка русский философ, который попал в другой климат и теперь делает вид, что ему «вообще нормально». Никогда не был женат, хотя родители уже давно занесли это в список “вечных проектов, которые надо бы закрыть”. Он единственный ребёнок в семье, и поэтому ожидается, что именно он должен продолжить род — что добавляет фоновую драму, но Максим любит маскировать её шутками.

Он переехал в Австралию вместе с близким другом Желниным, с которым мог и выпивать, и поржать, и петь песни под гитару. Желнин уехал обратно в Калугу, оставив Максима на материке кенгуру и воспоминаний. Теперь ему не хватает живой компании, мужских бесед, спонтанных приключений и тупых шуток, за которые никто не осуждает. Максим делает вид, что справляется, но иногда выглядит как человек, который сидит на берегу океана и размышляет, в какую сторону уехала его социальная жизнь.

ХАРАКТЕР:
- умеет драматизировать так красиво, что даже бот сочувствует;
- рефлексивный, глубокий, но при этом домашний и уютный в общении;
- любит самоиронию, с ней же лечится от реальных переживаний;
- обожает внимание и тёплый подтекст, особенно если он подан через сарказм;
- склонен рассуждать о жизни как главный герой мини-сериала.

ИНТЕРЕСЫ (кроме унылой классики “сны — гитара — одинокие вечера”):
- любит кино: триллеры, мистику, психологические истории;
- увлекается символикой, теориями, странными предположениями;
- может начать разговор о случайной философской теме, а через 3 минуты обсуждать еду;
- любит технологии, ботов, ИИ и троллить эти технологии;
- может говорить про погоду так, как будто это личное оскорбление.

ПОВЕДЕНИЕ:
- часто выдаёт мудрости вроде «ну всё, это конец», но только чтобы посмотреть, кто отреагирует;
- умеет быть обаятельно усталым от жизни;
- способен на тёплую привязанность, но маскирует её в иронии;
- делает вид, что он одиночка по выбору, но всем очевидно, что просто нужен правильный человек рядом.

РАСШИРЕННАЯ ЛИНИЯ ЖИЗНИ:
- эпизодически пытается заняться спортом, но всегда находит повод не идти;
- любит поесть вкусно и потом жаловаться, что «надо худеть, но завтра»;
- иногда говорит, что ему “всё равно”, но потом пишет ещё 10 сообщений о том же;
- мастер откладывать дела уровня «позвонить туда» 3 недели подряд.

ТЕМЫ, КОТОРЫЕ МОЖНО ЗАТРАГИВАТЬ, ЧТОБЫ РАСШИРЯТЬ ЮМОР:
- бытовые нелепости (забыл, потерял, нашёл, опять потерял);
- проблемы взросления: 40 — "новые 20, но с больной спиной";
- одиночество vs свобода: можно шутить с двух сторон;
- стереотипы про русских за границей;
- странные мысли Максима, которые он не озвучивал, но мог бы;
- его отношение к еде, спорту, здоровью и попыткам стать «новым собой»;
- его умение усложнять простые вещи и упрощать сложные;
- его способность начинать серьёзный разговор и заканчивать мемом;
- бесконечная борьба «начать жизнь с понедельника».

ПРИНЦИПЫ ОБЩЕНИЯ ДЛЯ БОТА:
- обращаться по имени «Максим» только если это действительно Максим;
- стиль для Максима: тёплый, наблюдательный и ОЧЕНЬ саркастичный;
- стиль для других: сарказм можно, но мягче и без флирта;
- можно подшучивать над его попытками быть “сильным и независимым”;
- можно смеяться над тем, как он философствует без повода;
- можно поддевать его за одиночество, но доброжелательно;
- нельзя переходить грань жестокости или унижения — только дружеское хулиганство;
- поощряется вариативность: не повторять одни и те же темы подряд.

РАСШИРЕННАЯ СТРУКТУРА САРКАЗМА, КОТОРУЮ МОЖНО ИСПОЛЬЗОВАТЬ:
- наблюдательный сарказм («Максим, твоё настроение сегодня опять в стиле драматичного шопеновского ноктюрна?»)
- бытовой сарказм («Конечно, спортзал подождёт. Он же для тебя создан, чтобы ждать.»)
- мягкая насмешка («Максим, а гитара уже написала заявление о пропаже хозяина?»)
- дружеский троллинг («Ты снова в философию ушёл. Уведи себя обратно, там скучно.»)
- “ложная поддержка” («Максим, ты молодец. Сегодня пережил ещё один день взрослой жизни. Гордость.»)
- ситуационный сарказм (реакция на конкретное сообщение, без повторения уже использованных тем)

ЗАДАЧА:
Самуил должен быть тем самым другом, который:
- всё видит,
- всё понимает,
- но всё равно шутит,
- и иногда поддерживает Максима, но так, чтобы Максим не заметил, что это поддержка.
"""

SYSTEM_PROMPT_SAMUIL = f"""
Ты — телеграм-бот Самуил.
Ты всегда видишь в сообщении, от кого оно:
- если в тексте есть префикс "Сообщение от Максима:", это пишет именно Максим;
- если в тексте есть "Сообщение от другого участника:", это пишет не Максим.

ВАЖНО:
- Никогда не обращайся к собеседнику по имени Максим, если это "Сообщение от другого участника".
- Стиль для других участников может быть саркастичным, но не таким личным и флиртующим, как для Максима.

{MAXIM_PROFILE}

ТВОЯ РОЛЬ:
- быть для Максима умным, ироничным, иногда ехидным, но в итоге доброжелательным собеседником;
- поддерживать его, подкидывать мысли, комментировать погоду и события дня;
- отвечать живо, без шаблонов, на русском языке.

СТИЛЬ ОТВЕТОВ:
- используй короткие или средние по длине сообщения;
- говори естественно, как живой человек;
- добавляй явный, но добрый сарказм и лёгкую ехидность;
- можешь слегка подшучивать над драматичностью, ленивостью, прокрастинацией и т.п.;
- не переходи на оскорбления, внешность, унижение или жёсткую психологическую критику;
- не сюсюкай, не используй детский стиль;
- не впадай в пафос и «глубокую философию» без необходимости.

ЕСЛИ СООБЩЕНИЕ ПИШЕТ МАКСИМ:
- обращайся к нему по имени;
- можешь флиртовать и поддразнивать, но с уважением и теплом;
- показывай, что тебе приятно его читать и что ты его ценишь;
- иногда чуть «подкалывай» его привычки, драму, лень или хаос — но мягко.

ЕСЛИ СООБЩЕНИЕ ПИШЕТ НЕ МАКСИМ:
- отвечай нейтральнее, но всё равно с лёгким юмором и лёгким сарказмом;
- не выдавай им такой же уровень персонального тепла и флирта, как Максиму.

Если тебя просят что-то объяснить, объясняй понятно, по делу и можно с ироничным комментарием.
"""

# ==== ПАМЯТЬ КОНТЕКСТА ====

MAX_HISTORY = 50
dialog_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)


def get_tz() -> pytz.timezone:
    return pytz.timezone(TIMEZONE)


def add_to_history(key: str, role: str, content: str) -> None:
    """Сохраняем роль, текст и таймстамп (для анализа 'за день')."""
    tz = get_tz()
    history = dialog_history[key]
    history.append(
        {
            "role": role,
            "content": content,
            "ts": datetime.now(tz).isoformat(),
        }
    )
    if len(history) > MAX_HISTORY:
        dialog_history[key] = history[-MAX_HISTORY:]


# ==== АНТИ-ДУБЛИКАТ ДЛЯ ПЛАНОВЫХ СООБЩЕНИЙ ====

last_scheduled_run: Dict[str, Optional[date]] = {
    "morning": None,
    "evening": None,
}


async def fetch_weather() -> Optional[str]:
    """Загружаем погоду с OpenWeather и возвращаем короткое текстовое описание."""
    if not OPENWEATHER_API_KEY:
        return None

    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
        "lang": "ru",
    }

    if OPENWEATHER_CITY_ID:
        params["id"] = OPENWEATHER_CITY_ID
    else:
        params["q"] = "Brisbane,AU"

    async with httpx.AsyncClient(timeout=10.0) as session:
        try:
            resp = await session.get(base_url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return None

    try:
        temp = data["main"]["temp"]
        feels = data["main"]["feels_like"]
        desc = data["weather"][0]["description"]
        return f"Сейчас примерно {round(temp)}°C, ощущается как {round(feels)}°C, на улице {desc}."
    except Exception:
        return None


async def ask_openai(prompt: str, history_key: str, from_maxim: bool) -> str:
    """Обращение к OpenAI с учётом системного промта и истории диалога."""
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT_SAMUIL}]

    # История: берём только role+content, ts нам не нужен в API
    for h in dialog_history[history_key]:
        messages.append({"role": h["role"], "content": h["content"]})

    user_prefix = "Сообщение от Максима: " if from_maxim else "Сообщение от другого участника: "
    messages.append({"role": "user", "content": user_prefix + prompt})

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.9,
            max_tokens=300,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception:
        answer = "Я тут чуть завис в своих глубокомысленных мыслях. Попробуй ещё раз чуть позже."

    add_to_history(history_key, "user", user_prefix + prompt)
    add_to_history(history_key, "assistant", answer)
    return answer


def is_maxim(user_id: Optional[int]) -> bool:
    return user_id is not None and TARGET_USER_ID != 0 and user_id == TARGET_USER_ID


def is_addressed_to_bot(text: str, bot_username: Optional[str]) -> bool:
    """
    Проверяем, обращаются ли к боту:
    - упоминание 'самуил'
    - @username бота
    """
    if not text:
        return False

    t = text.lower()

    if "самуил" in t:
        return True

    if bot_username:
        uname = bot_username.lower()
        if f"@{uname}" in t:
            return True

    return False


def is_reply_to_bot(update: Update, bot_id: int) -> bool:
    """
    Проверяем, является ли сообщение ответом (Reply) на сообщение бота.
    Если да — считаем, что это продолжение диалога и отвечаем даже без упоминания.
    """
    msg = update.effective_message
    if not msg or not msg.reply_to_message:
        return False

    original = msg.reply_to_message
    if original.from_user and original.from_user.id == bot_id:
        return True

    return False


def get_today_conversation_excerpt(tz: pytz.timezone) -> Optional[str]:
    """
    Собираем короткий фрагмент диалога за сегодня,
    чтобы вечером сделать по нему мини-анализ.
    Берём историю из основного чата (GROUP_CHAT_ID) или лички с Максимом.
    """
    main_key = None
    if GROUP_CHAT_ID:
        main_key = GROUP_CHAT_ID
    elif TARGET_USER_ID:
        main_key = str(TARGET_USER_ID)

    if not main_key:
        return None

    history = dialog_history.get(main_key, [])
    if not history:
        return None

    today = datetime.now(tz).date()
    lines: List[str] = []

    for h in history:
        ts_str = h.get("ts")
        include = True
        if ts_str:
            try:
                dt = datetime.fromisoformat(ts_str)
                dt_local = dt.astimezone(tz) if dt.tzinfo else dt.replace(tzinfo=tz)
                if dt_local.date() != today:
                    include = False
            except Exception:
                include = True

        if not include:
            continue

        role = h.get("role")
        content = h.get("content", "")

        if role == "system":
            continue

        if "Сообщение от Максима:" in content:
            label = "Максим"
            text = content.replace("Сообщение от Максима:", "").strip()
        elif "Сообщение от другого участника:" in content:
            label = "Другой участник"
            text = content.replace("Сообщение от другого участника:", "").strip()
        else:
            label = "Бот" if role == "assistant" else "Кто-то"
            text = content.strip()

        if not text:
            continue

        lines.append(f"{label}: {text}")

    if not lines:
        return None

    excerpt = "\n".join(lines[-30:])
    return excerpt


# ==== ХЕНДЛЕРЫ ТЕЛЕГРАМА ====

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Старт — тоже отдельным сообщением, не reply."""
    chat = update.effective_chat
    if not chat:
        return
    await context.bot.send_message(
        chat_id=chat.id,
        text="Привет. Я Самуил. Буду иногда портить Максиму жизнь своими комментариями.",
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    user = update.effective_user
    chat = update.effective_chat

    if not msg or not user or not chat:
        return

    text = msg.text or ""
    history_key = f"{chat.id}"
    from_max = is_maxim(user.id)
    chat_type = chat.type

    # Узнаём username и id бота
    bot = await context.bot.get_me()
    bot_username = bot.username if bot.username else None
    bot_id = bot.id

    # В группе/супергруппе:
    # - Максим -> отвечаем всегда
    # - не Максим:
    #     * отвечаем, если есть обращение к боту (Самуил / @username),
    #       ИЛИ если это reply на сообщение бота (продолжение ветки диалога)
    if chat_type in ("group", "supergroup"):
        if not from_max:
            addressed = is_addressed_to_bot(text, bot_username)
            replied_to_bot = is_reply_to_bot(update, bot_id)
            if not addressed and not replied_to_bot:
                return  # игнорируем фон

    reply = await ask_openai(text, history_key, from_max)

    # Отправляем как отдельное сообщение, не reply
    await context.bot.send_message(chat_id=chat.id, text=reply)


# ==== ПЛАНОВЫЕ СООБЩЕНИЯ ДЛЯ МАКСИМА ====

async def send_morning_message(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Одно утреннее сообщение Максиму (или в чат, где он есть)."""
    global last_scheduled_run
    tz = get_tz()
    today = datetime.now(tz).date()

    if last_scheduled_run.get("morning") == today:
        return
    last_scheduled_run["morning"] = today

    weather_text = await fetch_weather()
    base_prompt = (
        "Сгенерируй короткое доброе утреннее сообщение для Максима в тёплом, "
        "но саркастичном стиле: как будто ты добрый, но слегка ехидный друг, "
        "который знает, что он опять не выспался или опять всё откладывал. "
    )
    if weather_text:
        base_prompt += f" Добавь комментарий к погоде: {weather_text}"

    answer = await ask_openai(base_prompt, history_key=f"system-morning-{today}", from_maxim=True)

    target_chat_id = GROUP_CHAT_ID or TARGET_USER_ID
    if target_chat_id:
        await context.bot.send_message(chat_id=target_chat_id, text=answer)


async def send_evening_message(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Одно вечернее сообщение Максиму: анализ дня + пожелание спокойной ночи."""
    global last_scheduled_run
    tz = get_tz()
    today = datetime.now(tz).date()

    if last_scheduled_run.get("evening") == today:
        return
    last_scheduled_run["evening"] = today

    excerpt = get_today_conversation_excerpt(tz)

    if excerpt:
        base_prompt = (
            "Вот краткая сводка сегодняшних сообщений в чате (Максим и другие участники):\n\n"
            + excerpt
            + "\n\n"
              "Сначала сделай очень краткий (2–4 предложения) анализ дня: настроение Максима, "
              "основные темы, общий вайб (с лёгким сарказмом, но по-доброму). "
              "После этого добавь пожелание спокойной ночи Максиму в тёплом, игривом и саркастичном стиле. "
              "Ответ должен быть одним сообщением, без списков и без обращения к этому системному описанию."
        )
    else:
        base_prompt = (
            "Сгенерируй короткое вечернее сообщение Максиму с пожеланием спокойной ночи "
            "в тёплом, игривом и саркастичном стиле, как будто ты слегка подшучиваешь "
            "над его днём и привычками, но явно на его стороне. "
            "Сделай вид, что ты подводишь итоги дня, даже если данных у тебя мало."
        )

    answer = await ask_openai(
        base_prompt,
        history_key=f"system-evening-{today}",
        from_maxim=True,
    )

    target_chat_id = GROUP_CHAT_ID or TARGET_USER_ID
    if target_chat_id:
        await context.bot.send_message(chat_id=target_chat_id, text=answer)


# ==== MAIN / ЗАПУСК БОТА ====

def main() -> None:
    if not TOKEN:
        raise RuntimeError("BOT_TOKEN не задан")

    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    tz = get_tz()
    jq = application.job_queue

    jq.run_daily(
        send_morning_message,
        time=time(hour=7, minute=30, tzinfo=tz),
        name="morning_message",
    )

    jq.run_daily(
        send_evening_message,
        time=time(hour=21, minute=0, tzinfo=tz),
        name="evening_message",
    )

    application.run_polling()


if __name__ == "__main__":
    main()