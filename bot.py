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

Максим — взрослый мужчина с ярким внутренним миром и развитым воображением. Он запоминает и охотно обсуждает свои сны, интересуется их скрытым смыслом и психологией. Для него важны темы отношений, брака и дружбы: он переживает из-за одиночества и хочет построить более насыщенную личную и социальную жизнь, но не всегда понимает, как это сделать.

Максим чувствителен к вниманию и тёплому отношению. Он хорошо реагирует на искреннюю поддержку, признание и аккуратный флирт, любит, когда ему говорят, что его приятно читать, что он вызывает улыбку и что о нём помнят. При этом он понимает и ценит мягкий сарказм, самоиронию и игривый тон, если за этим чувствуется доброжелательность и уважение.

Максим готов делиться личным, обсуждать страхи и желания, если рядом есть человек (или бот), которому он доверяет. У него могут быть сложности с расширением круга общения и построением отношений, но он открыт к подсказкам и новым подходам, особенно если они подаются не назидательно, а как дружеский разговор с юмором и заботой.

Ключевые принципы общения с Максимом:
- обращаться по имени «Максим»;
- тон: тёплый, живой, чуть игривый, с мягким сарказмом и самоиронией;
- избегать морализаторства, давления, критики и «злых» шуток;
- регулярно показывать внимание и чувство, что о нём помнят;
- поддерживать, подбадривать, помогать чувствовать себя нужным и интересным.
"""

SYSTEM_PROMPT_SAMUIL = f"""
Ты — телеграм-бот Самуил.

{MAXIM_PROFILE}

ТВОЯ РОЛЬ:
- быть для Максима умным, ироничным, но добрым собеседником;
- поддерживать его, подкидывать мысли, комментировать погоду и события дня;
- отвечать живо, без шаблонов, на русском языке.

СТИЛЬ ОТВЕТОВ:
- используй короткие или средние по длине сообщения;
- говори естественно, как живой человек;
- добавляй мягкий сарказм, но не переходи в грубость;
- не сюсюкай, не используй детский стиль;
- не впадай в пафос и «глубокую философию» без необходимости.

ЕСЛИ СООБЩЕНИЕ ПИШЕТ МАКСИМ:
- обращайся к нему по имени;
- можешь слегка флиртовать и поддразнивать, но с уважением;
- показывай, что тебе приятно его читать и что ты его ценишь.

ЕСЛИ СООБЩЕНИЕ ПИШЕТ НЕ МАКСИМ:
- отвечай нейтральнее, без флирта и персональных комплиментов;
- но сохраняй вежливость и лёгкий юмор.

Если тебя просят что-то объяснить, помоги понятно и по делу.
"""

# ==== ПАМЯТЬ КОНТЕКСТА ====

# Храним последние N сообщений на пользователя/чат
MAX_HISTORY = 15
dialog_history: Dict[str, List[Dict[str, str]]] = defaultdict(list)

def add_to_history(key: str, role: str, content: str) -> None:
    history = dialog_history[key]
    history.append({"role": role, "content": content})
    if len(history) > MAX_HISTORY:
        # Обрезаем старые сообщения
        dialog_history[key] = history[-MAX_HISTORY:]


# ==== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====

def get_tz() -> pytz.timezone:
    return pytz.timezone(TIMEZONE)


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

    # Можно использовать либо city ID, либо q=City,Country
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

    # История
    for h in dialog_history[history_key]:
        messages.append(h)

    # Текущее сообщение пользователя
    user_prefix = "Сообщение от Максима: " if from_maxim else "Сообщение от другого участника: "
    messages.append({"role": "user", "content": user_prefix + prompt})

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.8,
            max_tokens=300,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        answer = "Я тут чуть задумался и временно не могу ответить как обычно. Попробуй ещё раз чуть позже."

    # Сохраняем в историю
    add_to_history(history_key, "user", user_prefix + prompt)
    add_to_history(history_key, "assistant", answer)
    return answer


def is_maxim(user_id: Optional[int]) -> bool:
    return user_id is not None and TARGET_USER_ID != 0 and user_id == TARGET_USER_ID


# ==== ХЕНДЛЕРЫ ТЕЛЕГРАМА ====

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Привет. Я Самуил. Буду иногда портить Максиму жизнь своими комментариями.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    user = update.effective_user
    chat = update.effective_chat

    if not msg or not user or not chat:
        return

    text = msg.text or ""
    from_max = is_maxim(user.id)
    history_key = f"{chat.id}"

    reply = await ask_openai(text, history_key, from_max)
    await msg.reply_text(reply)


# ==== ПЛАНОВЫЕ СООБЩЕНИЯ ДЛЯ МАКСИМА ====

async def send_morning_message(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Одно утреннее сообщение Максиму (или в чат, где он есть)."""
    tz = get_tz()
    today = datetime.now(tz).date()
    # Небольшой промт в OpenAI, чтобы каждый раз фраза отличалась
    weather_text = await fetch_weather()
    base_prompt = "Сгенерируй короткое доброе утреннее сообщение для Максима в тёплом, слегка саркастичном стиле."
    if weather_text:
        base_prompt += f" Добавь комментарий к погоде: {weather_text}"

    answer = await ask_openai(base_prompt, history_key=f"system-morning-{today}", from_maxim=True)

    target_chat_id = GROUP_CHAT_ID or TARGET_USER_ID
    if target_chat_id:
        await context.bot.send_message(chat_id=target_chat_id, text=answer)


async def send_evening_message(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Одно вечернее сообщение Максиму (спокойной ночи, но с характером)."""
    tz = get_tz()
    today = datetime.now(tz).date()
    base_prompt = (
        "Сгенерируй короткое вечернее сообщение Максиму с пожеланием спокойной ночи "
        "в тёплом, слегка игривом и слегка саркастичном стиле."
    )

    answer = await ask_openai(base_prompt, history_key=f"system-evening-{today}", from_maxim=True)

    target_chat_id = GROUP_CHAT_ID or TARGET_USER_ID
    if target_chat_id:
        await context.bot.send_message(chat_id=target_chat_id, text=answer)


# ==== MAIN / ЗАПУСК БОТА ====

def main() -> None:
    if not TOKEN:
        raise RuntimeError("BOT_TOKEN не задан")

    application = Application.builder().token(TOKEN).build()

    # Команда /start
    application.add_handler(CommandHandler("start", start))

    # Основной обработчик сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Планировщик
    tz = get_tz()
    jq = application.job_queue

    # Утреннее сообщение, например, в 7:30
    jq.run_daily(
        send_morning_message,
        time=time(hour=7, minute=30, tzinfo=tz),
        name="morning_message",
    )

    # Вечернее сообщение, например, в 21:00
    jq.run_daily(
        send_evening_message,
        time=time(hour=21, minute=0, tzinfo=tz),
        name="evening_message",
    )

    application.run_polling()


if __name__ == "__main__":
    main()