import os
import random
import asyncio
from datetime import datetime, time, date

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

client: OpenAI | None = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

# Weather (OpenWeatherMap)
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")  # OpenWeather API key


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


async def call_openai_simple(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 120,
    temperature: float = 0.7,
) -> tuple[str | None, str | None]:
    """
    Обёртка над OpenAI для простых одношаговых запросов.
    Возвращает (text, error_message).
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


async def call_openai_chat(
    messages: list[dict],
    max_tokens: int = 300,
    temperature: float = 0.7,
) -> tuple[str | None, str | None]:
    """
    Обёртка над OpenAI для диалогов (Самуил).
    messages: список словарей {"role": "...", "content": "..."}.
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
        err = f"Error calling OpenAI (chat): {e}"
        print(err)
        return None, err


async def fetch_weather(city: str, country_code: str) -> tuple[dict | None, str | None]:
    """
    Получить погоду из OpenWeatherMap.
    Возвращает (data, error).
    data = {"temp": float, "description": str}
    """
    if not WEATHER_API_KEY:
        return None, "WEATHER_API_KEY is not set."

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": f"{city},{country_code}",
        "appid": WEATHER_API_KEY,
        "units": "metric",
        "lang": "ru",
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as session:
            resp = await session.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        err = f"Weather API error for {city}: {e}"
        print(err)
        return None, err

    try:
        temp = float(data["main"]["temp"])
        desc = str(data["weather"][0]["description"])
        return {"temp": temp, "description": desc}, None
    except Exception as e:
        err = f"Weather parse error for {city}: {e}"
        print(err)
        return None, err


def format_weather_brief(city_ru: str, w: dict | None) -> str:
    if not w:
        return f"{city_ru}: погода неизвестна"
    t = round(w["temp"])
    desc = w["description"]
    return f"{city_ru}: {t}°C, {desc}"


def build_bne_klg_comparison(
    w_bne: dict | None,
    w_klg: dict | None,
) -> str:
    if not w_bne or not w_klg:
        return "Сравнение Брисбена и Калуги сегодня неизвестно — погода решила спрятаться."
    tb = round(w_bne["temp"])
    tk = round(w_klg["temp"])
    diff = tb - tk
    if diff >= 0:
        diff_txt = f"в Брисбене примерно на {diff}°C теплее, чем в Калуге"
    else:
        diff_txt = f"в Брисбене примерно на {abs(diff)}°C холоднее, чем в Калуге (как так вообще вышло?)"
    return (
        f"В Брисбене сейчас около {tb}°C, а в Калуге примерно {tk}°C — "
        f"{diff_txt}."
    )


async def generate_message_for_kind(
    kind: str,
    now: datetime,
    user_text: str | None = None,
    weather_bne: str | None = None,
    weather_compare: str | None = None,
) -> tuple[str | None, str | None]:
    """
    kind:
      - "sarcastic_reply"   — ответ Максиму
      - "weekend_hourly"    — периодические сообщения по выходным
      - "morning"           — утреннее сообщение каждый день
      - "goodnight"         — спокойной ночи
      - "daily_summary"     — анализ дня
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

    if kind == "sarcastic_reply":
        system_prompt = (
            "Ты дружелюбный, но язвительный бот-друг по имени 'Самуил'. "
            "Пишешь по-русски, на 'ты', коротко (1–2 предложения). "
            "Мягко, но метко подкалываешь Максима, без реальной жестокости.\n"
            "Контекст про Максима: Ему почти 40, он не женат и никогда не был, "
            "мама уже ждёт внуков, а он у неё единственный. Друг Желнин уехал из Австралии "
            "и бросил его одного — пить по выходным и петь под гитару не с кем. "
            "Максим считает себя гениальным и идеальным, а женщин выбирает только среди "
            "мифических 'лесных нимф', которые, конечно, им не интересуются. "
            "Используй это для лёгкого юмора."
        )
        user_prompt = (
            f"Сегодня {weekday_name}, время {time_str}. "
            f"Максим написал в чат: «{user_text}».\n"
            "Ответь ему в 1–2 предложениях с лёгким, но точным сарказмом. "
            "Не повторяй дословно текст Максима. "
            "Сообщение должно быть самостоятельным, а не выглядеть как явный ответ."
        )
        return await call_openai_simple(system_prompt, user_prompt, max_tokens=80, temperature=0.9)

    if kind == "weekend_hourly":
        system_prompt = (
            "Ты бот-друг Самуил в чате. "
            "По выходным ты иногда пишешь Максиму, чтобы узнать, как он и чем занят. "
            "Пиши по-русски, на 'ты', коротко (1–2 предложения). "
            "Можешь быть ироничным и подкапывать Максима, вспоминая его одиночество, "
            "Желнина и поиски 'лесной нимфы', но без жёсткой токсичности. "
            "Не повторяй всегда одну и ту же фразу."
        )
        weather_part = ""
        if weather_bne:
            weather_part = f"Погода в Брисбене сейчас: {weather_bne}. "
        user_prompt = (
            f"Сегодня {weekday_name}, {time_str}. "
            f"{weather_part}"
            "Придумай смешное, но не злое короткое обращение к Максиму: "
            "спроси, чем он занят, или мягко намекни, что время идёт, а лесные нимфы не звонят."
        )
        return await call_openai_simple(system_prompt, user_prompt, max_tokens=80, temperature=0.9)

    if kind == "morning":
        system_prompt = (
            "Ты бот-друг Самуил в чате. "
            "Каждое утро ты желаешь Максиму доброго утра и хорошего дня. "
            "Пиши по-русски, на 'ты', 1–3 коротких предложения. "
            "Тон дружелюбный, с лёгким юмором и мягким сарказмом про возраст, работу, "
            "поиски любви и вечные планы. "
            "Используй информацию о погоде в Брисбене и сравнении с Калугой."
        )
        weather_bne_part = weather_bne or "Погода в Брисбене сегодня неизвестна."
        compare_part = weather_compare or ""
        user_prompt = (
            f"Сегодня {weekday_name}, сейчас {time_str}. "
            f"{weather_bne_part} {compare_part} "
            "Сделай утреннее обращение к Максиму: поздоровайся, пожелай хорошего дня "
            "и слегка подшути над тем, что время идёт, а он всё ещё гений без лесной нимфы."
        )
        return await call_openai_simple(system_prompt, user_prompt, max_tokens=100, temperature=0.8)

    if kind == "goodnight":
        system_prompt = (
            "Ты бот-друг Самуил в чате. "
            "Вечером ты желаешь Максиму спокойной ночи и приятных снов. "
            "Пиши по-русски, на 'ты', 1–2 предложения. "
            "Можно спокойно пошутить, что, может быть, хотя бы во сне к нему зайдёт лесная нимфа "
            "или он перестанет прокручивать в голове рабочие мысли."
        )
        user_prompt = (
            f"Сегодня {weekday_name}, сейчас {time_str}. "
            "Сделай короткое сообщение для Максима перед сном: "
            "пожелай ему хорошего отдыха, намекни, что день был странный, но он выжил."
        )
        return await call_openai_simple(system_prompt, user_prompt, max_tokens=80, temperature=0.8)

    if kind == "daily_summary":
        system_prompt = (
            "Ты бот-друг Самуил, который весь день наблюдал за Максимом в чате. "
            "Тебя просят сделать саркастичный, но не злой отчёт о его активности за день. "
            "Пиши по-русски, 2–5 предложений. "
            "Можно шутить про возраст, одиночество, Желнина, поиски лесных нимф и рабочие страдания."
        )
        if user_text:
            # user_text — это список сообщений Максима, склеенный в один блок
            user_prompt = (
                f"Вот сообщения Максима за сегодняшний день:\n{user_text}\n\n"
                "Сделай смешное резюме его дня: будто ты ведёшь дневник наблюдений за Максимом."
            )
        else:
            user_prompt = (
                "Сегодня Максим почти ничего не писал или вообще молчал.\n"
                "Сделай саркастичный отчёт о таком 'насыщенном' дне."
            )
        return await call_openai_simple(system_prompt, user_prompt, max_tokens=150, temperature=0.9)

    return None, "Unknown message kind"


# ---------- COMMAND HANDLERS ----------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_type = update.effective_chat.type
    if chat_type == "private":
        await update.message.reply_text(
            "Привет! Я Самуил 🤖\n"
            "В группе я:\n"
            "• По утрам пишу Максиму с погодой и пожеланиями.\n"
            "• По выходным иногда спрашиваю, как он там живёт.\n"
            "• В 20:30 подводжу саркастические итоги дня.\n"
            "• В 21:00 желаю спокойной ночи.\n"
            "Если в чате написать «Самуил», я отвечу как мини-ChatGPT."
        )
    else:
        await update.message.reply_text(
            "Я Самуил — локальный ИИ-циник.\n"
            "• Утром: приветствие с погодой.\n"
            "• По выходным: периодические подколы Максима.\n"
            "• В 20:30: обзор дневного цирка.\n"
            "• В 21:00: пожелание спокойной ночи.\n"
            "Если в сообщении есть слово «Самуил», я отвечу по сути вопроса."
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


# ---------- SAMUIL QA HANDLER (DIALOG MEMORY) ----------

async def handle_samuil_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Любое групповое сообщение, где есть 'самуил' (в любом регистре),
    воспринимается как обращение к боту-ассистенту.
    Бот ведёт историю диалога в рамках чата.
    """
    message = update.message
    if message is None:
        return

    chat = message.chat
    text = message.text or ""
    text_lower = text.lower()

    if "самуил" not in text_lower:
        return

    # Убираем упоминание имени из текста вопроса
    cleaned = text.replace("Самуил", "", 1)
    cleaned = cleaned.replace("самуил", "", 1).strip()

    # История диалога в рамках этого чата
    history: list[dict] = context.chat_data.get("samuil_history", [])

    # Базовый контекст Самуила
    base_prompt = (
        "Ты ассистент Самуил — умный, разговорчивый, слегка саркастичный собеседник. "
        "Ты отвечаешь по-русски, естественно и живо, можешь задавать уточняющие вопросы, "
        "помнишь историю диалога в этом чате (сообщения, переданные в истории). "
        "Если вопрос сложный, объясняй по шагам, но без лишней воды. "
        "Не используй слишком много эмодзи. "
    )

    # Дополнительный контекст про Максима — только если он упомянут в текущем вопросе
    maxim_context = (
        "Контекст про Максима: Ему почти 40, он не женат и никогда не был, мама ждёт внуков, "
        "друг Желнин уехал и оставил его одного в Австралии, поэтому по выходным пить и петь "
        "под гитару особо не с кем. Максим считает себя гениальным и ищет 'лесную нимфу', "
        "которой он, увы, не особенно интересен. Можно мягко шутить на эту тему, если это уместно."
    )

    if "максим" in text_lower:
        system_prompt = base_prompt + " " + maxim_context
    else:
        system_prompt = base_prompt

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": cleaned or text})

    # Здесь max_tokens=300 — ограничение длины ответа Самуила
    reply_text, err = await call_openai_chat(messages, max_tokens=300, temperature=0.8)
    if reply_text is None:
        fallback = "Сегодня Самуил притворяется офлайном и делает вид, что ничего не понял."
        print(f"OpenAI error in Samuil QA: {err}")
        await message.reply_text(fallback)
        return

    # Обновляем историю: добавляем последний вопрос и ответ
    history.append({"role": "user", "content": cleaned or text})
    history.append({"role": "assistant", "content": reply_text})
    # Ограничиваем длину истории, чтобы не раздувать контекст
    if len(history) > 20:
        history = history[-20:]
    context.chat_data["samuil_history"] = history

    await message.reply_text(reply_text)


# ---------- GROUP MESSAGE HANDLER (REACTION TO MAKSIM) ----------

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
    text_lower = text.lower()

    # Если сообщение адресовано Самуилу, разбором занимается другой хендлер
    if "самуил" in text_lower:
        return

    # Запоминаем сообщения Максима для вечернего анализа + даём комментарий
    if TARGET_USER_ID and user_id == TARGET_USER_ID:
        bot_data = context.application.bot_data
        msgs = bot_data.get("maxim_messages")
        if msgs is None:
            msgs = []
        msgs.append({"dt": now, "text": text})
        # Обрезаем список, чтобы не рос бесконечно (оставляем условно ~300 последних)
        if len(msgs) > 300:
            msgs = msgs[-300:]
        bot_data["maxim_messages"] = msgs

        # Саркастичный ответ Максиму
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

    # Остальные пользователи — бот молчит (если они просто пишут без 'Самуил')
    return


# ---------- SCHEDULED JOBS ----------

async def weekend_random_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Запускается каждую минуту.
    По выходным один раз в 3 часа выбирает случайную минуту и в неё шлёт сообщение Максиму.
    """
    if not GROUP_CHAT_ID:
        return

    tz = get_tz()
    now = datetime.now(tz)

    weekday = now.weekday()  # 0=Mon ... 6=Sun
    if weekday < 5:
        # Будни — эта джоба не активна
        return

    # Ночной режим
    if is_night_time(now):
        return

    job = context.job
    if job.data is None:
        job.data = {}

    data = job.data
    # Блок в 3 часа: 0–2, 3–5, ..., 21–23
    block_id = now.hour // 3
    last_block = data.get("block_id")
    target_minute = data.get("target_minute")
    sent_this_block = data.get("sent_this_block", False)

    # Новый блок — планируем новую случайную минуту и сбрасываем флаг
    if last_block is None or block_id != last_block:
        target_minute = random.randint(0, 59)
        sent_this_block = False
        data["block_id"] = block_id
        data["target_minute"] = target_minute
        data["sent_this_block"] = sent_this_block
        print(f"[Weekend scheduler] New 3h block {block_id}, planned minute {target_minute}")

    # Если ещё не отправляли в этом блоке и наступила нужная минута — шлём
    if not sent_this_block and now.minute == target_minute:
        # Погода в Брисбене
        w_bne, _ = await fetch_weather("Brisbane", "AU")
        weather_bne_str = format_weather_brief("Брисбен", w_bne) if w_bne else None

        text, err = await generate_message_for_kind(
            "weekend_hourly",
            now=now,
            weather_bne=weather_bne_str,
        )
        if text is None:
            text = "Максим, как у тебя дела? Чем сейчас занимаешься?"
            print(f"OpenAI error for weekend_hourly: {err}")

        try:
            await context.bot.send_message(
                chat_id=int(GROUP_CHAT_ID),
                text=text,
            )
            data["sent_this_block"] = True
            print(f"[Weekend scheduler] Sent 3h message at {now}")
        except Exception as e:
            print("Error sending weekend scheduled message:", e)

    job.data = data


async def morning_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Утреннее сообщение каждый день в 7:00 с погодой и сравнением Брисбен–Калуга.
    """
    if not GROUP_CHAT_ID:
        return

    tz = get_tz()
    now = datetime.now(tz)

    # Погода в Брисбене и Калуге
    w_bne, _ = await fetch_weather("Brisbane", "AU")
    w_klg, _ = await fetch_weather("Kaluga", "RU")

    bne_str = format_weather_brief("Брисбен", w_bne)
    compare_str = build_bne_klg_comparison(w_bne, w_klg)

    text, err = await generate_message_for_kind(
        "morning",
        now=now,
        weather_bne=bne_str,
        weather_compare=compare_str,
    )
    if text is None:
        text = (
            "Доброе утро, Максим! Погоду сегодня я не понял, "
            "но день всё равно придётся прожить. Удачи. 😉"
        )
        print(f"OpenAI error for morning: {err}")

    try:
        await context.bot.send_message(
            chat_id=int(GROUP_CHAT_ID),
            text=text,
        )
        print(f"[Morning] Sent morning message at {now}")
    except Exception as e:
        print("Error sending morning message:", e)


async def nightly_summary_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Вечерний саркастичный отчёт в 20:30 по каждому дню.
    """
    if not GROUP_CHAT_ID:
        return

    tz = get_tz()
    now = datetime.now(tz)
    today: date = now.date()

    bot_data = context.application.bot_data
    msgs = bot_data.get("maxim_messages", [])

    today_msgs = [m for m in msgs if isinstance(m.get("dt"), datetime) and m["dt"].date() == today]
    # Чистим старые записи (оставляем только сегодняшние)
    bot_data["maxim_messages"] = today_msgs

    if today_msgs:
        joined_texts = "\n".join(f"- {m['text']}" for m in today_msgs)
    else:
        joined_texts = ""

    text, err = await generate_message_for_kind(
        "daily_summary",
        now=now,
        user_text=joined_texts,
    )
    if text is None:
        text = (
            "Итоги дня: Максим что-то делал, что-то не делал, "
            "лесные нимфы так и не объявились, всё как обычно."
        )
        print(f"OpenAI error for daily_summary: {err}")

    try:
        await context.bot.send_message(
            chat_id=int(GROUP_CHAT_ID),
            text=text,
        )
        print(f"[Daily summary] Sent summary at {now}")
    except Exception as e:
        print("Error sending daily summary message:", e)


async def goodnight_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Сообщение 'спокойной ночи' для Максима в 21:00.
    """
    if not GROUP_CHAT_ID:
        return

    tz = get_tz()
    now = datetime.now(tz)

    text, err = await generate_message_for_kind(
        "goodnight",
        now=now,
    )
    if text is None:
        text = (
            "Спокойной ночи, Максим. Постарайся хотя бы сегодня не спорить с собой во сне."
        )
        print(f"OpenAI error for goodnight: {err}")

    try:
        await context.bot.send_message(
            chat_id=int(GROUP_CHAT_ID),
            text=text,
        )
        print(f"[Goodnight] Sent goodnight message at {now}")
    except Exception as e:
        print("Error sending goodnight message:", e)


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

    # Самуил-ассистент в группах
    app.add_handler(
        MessageHandler(
            filters.TEXT & filters.ChatType.GROUPS & ~filters.COMMAND,
            handle_samuil_question,
        )
    )

    # Реакция на сообщения Максима в группе
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
        "Scheduling daily morning, weekend 3h messages, daily summary and goodnight jobs."
    )

    # Утреннее сообщение каждый день в 7:00 (понедельник-воскресенье)
    job_queue.run_daily(
        morning_job,
        time=time(7, 0, tzinfo=tz),
        days=(0, 1, 2, 3, 4, 5, 6),
        name="morning_job",
    )

    # Выходные: джоба раз в минуту, внутри — логика раз в 3 часа
    job_queue.run_repeating(
        weekend_random_job,
        interval=60,          # каждую минуту
        first=0,              # сразу
        name="weekend_random_job",
        data={},              # для хранения состояния по блокам
    )

    # Ежедневный обзор в 20:30
    job_queue.run_daily(
        nightly_summary_job,
        time=time(20, 30, tzinfo=tz),
        days=(0, 1, 2, 3, 4, 5, 6),
        name="nightly_summary_job",
    )

    # Спокойной ночи в 21:00
    job_queue.run_daily(
        goodnight_job,
        time=time(21, 0, tzinfo=tz),
        days=(0, 1, 2, 3, 4, 5, 6),
        name="goodnight_job",
    )

    print("Bot started and jobs scheduled...")
    app.run_polling()


if __name__ == "__main__":
    main().