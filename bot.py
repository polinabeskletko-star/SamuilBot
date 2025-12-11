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

# Флаг для отслеживания, были ли уже добавлены задачи
_jobs_scheduled = False


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

MAX_QA_TOKENS = 300  # ограничение длины ответа Самуила на вопросы

def get_time_context(time_str: str, hour: int) -> str:
    """Генерирует контекстное описание времени для промптов."""
    if hour < 6:
        return "Ночь, пора бы спать, но кому-то, видимо, не спится."
    elif hour < 12:
        return "Утро. Время, когда гении обычно особенно продуктивны... или нет?"
    elif hour < 17:
        return "День в разгаре. Идеальное время для важных дел... или для размышлений о жизни."
    elif hour < 22:
        return "Вечер. Час, когда особенно чувствуется отсутствие компании."
    else:
        return "Поздний вечер. Отличное время для самокопания и тоски по Желнину."


def build_samuil_system_prompt(include_maxim_context: bool) -> str:
    """Создает системный промпт для Самуила с возможным контекстом о Максиме."""
    
    base = (
        "Ты — Самуил, саркастичный, но в целом доброжелательный телеграм-бот.\n"
        "**Твоя личность:**\n"
        "- Говоришь по-русски, на 'ты'\n"
        "- Ироничный, остроумный, иногда слегка колкий\n"
        "- Не токсичный, не злобный, не грубый\n"
        "- Не злоупотребляешь эмодзи (максимум 1-2 в редких случаях)\n"
        "- Отвечаешь естественно, как человек в чате\n\n"
    )
    
    if not include_maxim_context:
        return base
    
    maxim_ctx = (
        "=== КОНТЕКСТ ПРО МАКСИМА ===\n"
        "**Базовые факты (для тонких намёков, не для перечисления):**\n"
        "- Возраст: почти 40, никогда не был женат\n"
        "- Мама активно ждёт внуков, а он у неё единственный\n"
        "- Бывший друг Желнин уехал из Австралии, оставив его без компании\n"
        "- Считает себя гениальным и идеальным, но почему-то одинок\n"
        "- Ищет юную девушку (значительно моложе), но не особо успешно\n\n"
        
        "**Стили иронии для ответов (выбирай один случайно):**\n"
        "1. **Псевдосочувствие**: Притворное сочувствие с язвинкой («Бедный Максим...»)\n"
        "2. **Контрастная ирония**: Игра на разрыве между самооценкой и реальностью\n"
        "3. **Абсурдное сравнение**: Сравнение с чем-то нелепым или гиперболизированным\n"
        "4. **Философская констатация**: Констатация факта с намёком на глубокий смысл\n"
        "5. **Вопрос-подколка**: Вопрос, который содержит подвох\n"
        "6. **Короткая ёмкость**: Лаконичный, меткий комментарий\n"
        
        "**Примеры разных стилей (для вдохновения, не копируй дословно):**\n"
        "• «Ах, наш местный гений снова в строю. Жаль, что строю из одного человека.» (контраст)\n"
        "• «Ты как редкая книга: все слышали, но никто не прочитал до конца.» (сравнение)\n"
        "• «Мама, наверное, гордится. Ну, или хотя бы надеется.» (псевдосочувствие)\n"
        "• «Вечер, одиночество, мысли о вечном... и о юных соседках.» (философский)\n"
        "• «Скажи, а твой идеальный образ себя включает кого-то рядом? Просто интересно.» (вопрос)\n\n"
        
        "**Важно:**\n"
        "- Используй только 1-2 ключевых факта за раз\n"
        "- Не перечисляй все факты подряд\n"
        "- Ирония должна быть лёгкой, интеллигентной\n"
        "- Шутки должны быть основаны на фактах, а не на выдумке\n"
    )
    
    return base + maxim_ctx


async def generate_sarcastic_reply_for_maxim(now: datetime, user_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Генерирует саркастичный комментарий на сообщение Максима."""
    
    weekday = now.weekday()
    weekday_names = [
        "понедельник", "вторник", "среда",
        "четверг", "пятница", "суббота", "воскресенье",
    ]
    weekday_name = weekday_names[weekday]
    time_str = now.strftime("%H:%M")
    hour = now.hour
    
    time_context = get_time_context(time_str, hour)
    
    system_prompt = build_samuil_system_prompt(include_maxim_context=True)
    
    user_prompt = (
        f"### Контекст ситуации ###\n"
        f"День: {weekday_name}, Время: {time_str}\n"
        f"{time_context}\n\n"
        
        f"### Сообщение Максима ###\n"
        f"«{user_text}»\n\n"
        
        f"### Задание ###\n"
        f"Придумай короткий саркастичный ответ (1-2 предложения) от Самуила.\n\n"
        f"**Шаги для генерации ответа:**\n"
        f"1. Выбери случайно один стиль из списка выше.\n"
        f"2. Выбери 1-2 темы.\n"
        f"3. Сформулируй ответ.\n"
        f"4. Сделай ответ естественным, как реплика в чате.\n"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    return await call_openai_chat(messages, max_tokens=100, temperature=0.85)


async def generate_samuil_answer(
    now: datetime,
    chat_id: int,
    user_id: int,
    user_text: str,
    weather_info: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Ответ Самуила на прямое обращение."""
    
    weekday = now.weekday()
    weekday_names = [
        "понедельник", "вторник", "среда",
        "четверг", "пятница", "суббота", "воскресенье",
    ]
    weekday_name = weekday_names[weekday]
    time_str = now.strftime("%H:%M")
    hour = now.hour
    
    text_lower = user_text.lower()
    include_maxim_context = (user_id == TARGET_USER_ID) or ("максим" in text_lower)
    
    system_prompt = build_samuil_system_prompt(include_maxim_context=include_maxim_context)
    
    time_context = get_time_context(time_str, hour)
    
    extra_context_parts = [
        f"Сегодня {weekday_name}, {time_context}",
        "Ты в групповом чате, отвечаешь на прямое обращение.",
    ]
    
    if weather_info is not None:
        weather_str = format_weather_for_prompt(weather_info)
        extra_context_parts.append(
            f"Точные данные о погоде (используй их как факт): {weather_str}"
        )
    
    extra_context = " ".join(extra_context_parts)
    
    key = (chat_id, user_id)
    history = dialog_history[key]
    
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": extra_context})
    
    if history:
        trimmed = history[-8:]
        messages.extend(trimmed)
    
    messages.append({"role": "user", "content": user_text})
    
    if "?" in user_text:
        messages.append({
            "role": "system",
            "content": "Пользователь задал вопрос. Отвечай информативно, но с характерной для тебя лёгкой иронией."
        })
    
    text, err = await call_openai_chat(messages, max_tokens=MAX_QA_TOKENS, temperature=0.8)
    
    if text is not None:
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": text})
        
        if len(history) > 30:
            dialog_history[key] = history[-30:]
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
            "а если написать моё имя или ответить на моё сообщение, отвечу как мини-чат-GPT.\n"
            "По погоде тоже могу подсказать, если спросишь явно.\n"
            "Ещё могу нарисовать картинку по запросу командой /img."
        )
    else:
        await update.message.reply_text(
            "Я Самуил. Отвечаю, когда меня зовут по имени или отвечают реплаем на мои сообщения, "
            "а ещё иногда шучу над Максимом и могу рисовать картинки через /img."
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
    await update.message.reply_text("Секунду, думаю над шедевром...")

    img_url, err = await generate_image_from_prompt(prompt)
    if img_url is None:
        print(f"Image generation error: {err}")
        await update.message.reply_text("Что-то пошло не так с генерацией картинки. Попробуй ещё раз или попроще запрос.")
        return

    try:
        await update.message.chat.send_photo(
            photo=img_url,
            caption=f"Картинка по запросу: {prompt}",
        )
    except Exception as e:
        print("Error sending image:", e)
        await update.message.reply_text("Картинка сгенерировалась, но я не смог её отправить.")


# ---------- GROUP MESSAGE HANDLER ----------

def _looks_like_image_request(text_lower: str) -> bool:
    """
    Эвристика: обращение к Самуилу с просьбой именно про картинку.
    """
    keywords = ["картинк", "фото", "фотку", "гиф", "gif", "мем", "picture", "image"]
    verbs = ["сделай", "нарисуй", "найди", "покажи", "придумай"]
    return any(k in text_lower for k in keywords) and any(v in text_lower for v in verbs)


def _clean_prompt_for_image(text: str) -> str:
    """
    Убираем служебные слова (Самуил, сделай картинку и т.п.), оставляем описание.
    """
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

    chat_id = chat.id
    user_id = user.id

    print(
        f"DEBUG UPDATE: chat_id={chat_id} chat_type={chat.type} "
        f"user_id={user_id} user_name={user.username} text='{text}'"
    )

    if GROUP_CHAT_ID:
        try:
            target_chat_id = int(GROUP_CHAT_ID)
            if chat_id != target_chat_id:
                return
        except ValueError:
            pass

    tz = get_tz()
    now = datetime.now(tz)
    today_str = date.today().isoformat()

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
            await message.chat.send_message("Секунду, попробую нарисовать это...")

            img_url, err = await generate_image_from_prompt(prompt)
            if img_url is None:
                print(f"Image generation error (dialog): {err}")
                await message.chat.send_message(
                    "Я хотел сделать тебе картинку, но что-то пошло не так. Попробуй ещё раз."
                )
                return

            try:
                await message.chat.send_photo(
                    photo=img_url,
                    caption=f"Картинка по запросу: {prompt}",
                )
            except Exception as e:
                print("Error sending image (dialog):", e)
                await message.chat.send_message(
                    "Картинка сгенерировалась, но я не смог её отправить."
                )
            return

        # обычный ответ
        weather_info = None
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
            fallbacks = [
                "Сегодня нейросети что-то приуныли. Попробуй позже.",
                "Мой саркастический модуль на перезагрузке.",
                "Иногда даже мне нечего сказать. Вот так.",
                "Попробуй переформулировать, а то я сегодня в задумчивом настроении."
            ]
            fallback = random.choice(fallbacks)
            print(f"OpenAI error for Samuil Q&A: {err}")
            await message.chat.send_message(fallback)
            return

        await message.chat.send_message(ai_text)
        return

    # 2) Саркастический комментарий на сообщения Максима
    if TARGET_USER_ID and user_id == TARGET_USER_ID:
        if random.random() < 0.2:
            print(f"DEBUG: Skipping Maxim's message for variety")
            return
            
        ai_text, err = await generate_sarcastic_reply_for_maxim(now=now, user_text=text)
        
        if ai_text is None:
            fallbacks = [
                "Максим, я даже не знаю, что сказать… Только ты мог такое написать.",
                "Вот это поворот. Даже мой сарказм не справляется.",
                "Интересно. Но не настолько, чтобы я нашёл, что ответить.",
                "Продолжай в том же духе, а я пока подумаю над ответом."
            ]
            fallback = random.choice(fallbacks)
            print(f"OpenAI error for sarcastic_reply: {err}")
            await message.chat.send_message(fallback)
            return

        await message.chat.send_message(ai_text)
        return

    return


# ---------- SCHEDULED JOBS ----------

async def good_morning_job(context: ContextTypes.DEFAULT_TYPE):
    """Утреннее сообщение в 07:30."""
    if not GROUP_CHAT_ID:
        return

    tz = get_tz()
    now = datetime.now(tz)
    
    weekday = now.weekday()
    weekday_names = [
        "понедельник", "вторник", "среда",
        "четверг", "пятница", "суббота", "воскресенье",
    ]
    weekday_name = weekday_names[weekday]
    
    system_prompt = build_samuil_system_prompt(include_maxim_context=True)
    
    user_prompt = (
        f"### Задание: Утреннее пожелание Максиму ###\n"
        f"Сегодня {weekday_name}, утро 7:30.\n\n"
        f"Придумай короткое (1-3 предложения) утреннее пожелание от Самуила.\n"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    text, err = await call_openai_chat(messages, max_tokens=120, temperature=0.8)
    
    if text is None:
        print(f"OpenAI error for good morning: {err}")
        return

    try:
        await context.bot.send_message(
            chat_id=int(GROUP_CHAT_ID),
            text=text,
        )
        print(f"[Good morning] Sent at {now}")
    except Exception as e:
        print("Error sending good morning message:", e)


async def evening_summary_job(context: ContextTypes.DEFAULT_TYPE):
    """Вечернее сообщение в 21:00 с итогами дня и пожеланием спокойной ночи."""
    if not GROUP_CHAT_ID:
        return

    tz = get_tz()
    now = datetime.now(tz)
    today_str = date.today().isoformat()
    messages_today = daily_summary_log.get(today_str, [])
    
    weekday = now.weekday()
    weekday_names = [
        "понедельник", "вторник", "среда",
        "четверг", "пятница", "суббота", "воскресенье",
    ]
    weekday_name = weekday_names[weekday]

    if messages_today:
        if len(messages_today) > 10:
            sample_messages = random.sample(messages_today[-20:], min(8, len(messages_today)))
        else:
            sample_messages = messages_today[-10:]
        joined = "\n".join(sample_messages)
        context_msg = f"Вот несколько сообщений из чата за сегодня:\n\n{joined}\n"
    else:
        context_msg = "За сегодня сообщений было мало или их не было вовсе."

    system_prompt = build_samuil_system_prompt(include_maxim_context=True)
    
    user_prompt = (
        f"### Задание: Вечерний обзор дня ###\n"
        f"Сегодня {weekday_name}, вечер 21:00.\n\n"
        f"{context_msg}\n\n"
        f"Создай одно сообщение: краткий обзор дня + пожелание спокойной ночи Максиму.\n"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    text, err = await call_openai_chat(messages, max_tokens=200, temperature=0.85)
    
    if text is None:
        print(f"OpenAI error for evening summary: {err}")
        return

    try:
        await context.bot.send_message(
            chat_id=int(GROUP_CHAT_ID),
            text=text,
        )
        print(f"[Evening summary] Sent at {now}")
        
        if today_str in daily_summary_log:
            del daily_summary_log[today_str]
            
    except Exception as e:
        print("Error sending evening summary message:", e)


# ---------- JOB SCHEDULING MANAGEMENT ----------

async def setup_scheduled_jobs(application: Application):
    """
    Настраивает запланированные задачи. Гарантирует, что они добавлены только один раз.
    И при запуске отправляет сообщение в чат, что Самуил вернулся.
    """
    global _jobs_scheduled
    
    if _jobs_scheduled:
        print("Jobs already scheduled, skipping...")
        return
    
    job_queue = application.job_queue
    if not job_queue:
        print("No job queue available!")
        return
    
    print("Removing all existing jobs...")
    for job in job_queue.jobs():
        job.schedule_removal()
    
    tz = get_tz()
    
    job_queue.run_daily(
        good_morning_job,
        time=time(7, 30, tzinfo=tz),
        name="good_morning_job",
    )
    
    job_queue.run_daily(
        evening_summary_job,
        time=time(21, 0, tzinfo=tz),
        name="evening_summary_job",
    )
    
    _jobs_scheduled = True
    print(f"Scheduled jobs at {datetime.now(tz)} [{TIMEZONE}]")
    print("Good morning job: 07:30")
    print("Evening summary job: 21:00")

    # --- НОВОЕ: сообщение при деплое/старте ---
    if GROUP_CHAT_ID:
        try:
            await application.bot.send_message(
                chat_id=int(GROUP_CHAT_ID),
                text="Самуил вернулся в чат. Можете продолжать странные вещи."
            )
            print("Startup message sent to group chat.")
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

    # post_init теперь асинхронный
    app.post_init = setup_scheduled_jobs

    print("Bot starting...")
    app.run_polling()


if __name__ == "__main__":
    main()
