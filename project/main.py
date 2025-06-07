from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel, EmailStr
from experta import KnowledgeEngine, Rule, Fact
from tech_recommendation import TechRecommendationEngine
import sqlite3
import psycopg2
import json
import smtplib
from psycopg2 import IntegrityError
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import numpy as np
import random
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi.middleware.cors import CORSMiddleware
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib import colors
import io
import os
import spacy
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# conn = sqlite3.connect("tech_stack.db", check_same_thread=False)
conn = psycopg2.connect(
    dbname="tech_stack",
    user="postgres",  
    password="3932323",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# Загрузка языковых моделей
try:
    nlp = spacy.load("ru_core_news_md")
except:
    # Если модель не установлена, предложить пользователю установить ее
    print("Модель ru_core_news_md не найдена. Установите ее командой: python -m spacy download ru_core_news_md")
    # Временно используем небольшую модель
    nlp = spacy.load("ru_core_news_sm")

# Инициализация BERT модели для русского языка
try:
    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")
except:
    print("Не удалось загрузить модель BERT. Убедитесь, что установлен пакет transformers и есть доступ к интернету.")
    # Создадим заглушки для функций эмбеддинга на случай, если модель не загрузилась
    class DummyModel:
        def __call__(self, **kwargs):
            class DummyOutput:
                def __init__(self):
                    self.last_hidden_state = torch.zeros((1, 1, 768))
            return DummyOutput()
    
    class DummyTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.zeros((1, 10)), "attention_mask": torch.zeros((1, 10))}
    
    tokenizer = DummyTokenizer()
    model = DummyModel()

class UserMessage(BaseModel):
    message: str
    user_id: str

class EmailRequest(BaseModel):
    user_email: EmailStr
    user_id: str

class NaturalDialogManager:
    def __init__(self):
        # Определение структуры диалога
        self.dialog_flow = {
            "start": {
                "question": [
                    "Давайте поговорим о типе приложения, которое вы хотите разработать. Это будет нативное, кроссплатформенное или гибридное приложение?",
                    "Какой тип приложения вас интересует? Нативное обычно более производительное, кроссплатформенное проще в разработке, а гибридное сочетает оба подхода.",
                    "Для начала, расскажите, какой тип приложения вы планируете? Нативное, кроссплатформенное или гибридное?"
                ],
                "options": ["нативное", "кроссплатформенное", "гибридное"],
                "next": lambda x: "platform" if x == "нативное" else "experience_level"
            },
            "platform": {
                "question": [
                    "Для какой платформы будем разрабатывать? iOS или Android?",
                    "На какой платформе должно работать ваше приложение? iOS или Android?",
                    "Скажите, вы нацелены на iOS или Android устройства?"
                ],
                "options": ["ios", "android"],
                "next": "experience_level"
            },
            "experience_level": {
                "question": [
                    "Как бы вы оценили уровень опыта вашей команды разработчиков? Новичок, средний или эксперт?",
                    "Расскажите о навыках вашей команды. Они новички, имеют средний опыт или являются экспертами?",
                    "Насколько опытна ваша команда в разработке? Это новички, разработчики среднего уровня или эксперты?"
                ],
                "options": ["новичок", "средний", "эксперт"],
                "next": "performance"
            },
            "performance": {
                "question": [
                    "Насколько важна производительность для вашего приложения? Она критична, средней важности или не критична?",
                    "Поговорим о производительности. Она критично важна для вашего проекта, имеет среднюю важность или не так критична?",
                    "Как бы вы оценили важность высокой производительности для вашего приложения? Критична, средняя или не критична?"
                ],
                "options": ["критична", "средняя", "не критична"],
                "next": "speed"
            },
            "speed": {
                "question": [
                    "Как насчет скорости разработки? Вам нужна быстрая, средняя или вы готовы к более длительной разработке?",
                    "Какие у вас сроки разработки? Нужна быстрая реализация, средние темпы или можно не спешить?",
                    "Расскажите о ваших временных ограничениях. Разработка должна быть быстрой, средней или низкой скорости?"
                ],
                "options": ["быстрая", "средняя", "низкая"],
                "next": "cost"
            },
            "cost": {
                "question": [
                    "И последний вопрос - каков ваш бюджет на разработку? Низкий, средний или высокий?",
                    "Давайте обсудим финансовую сторону. Ваш бюджет на проект низкий, средний или высокий?",
                    "Какими финансовыми ресурсами вы располагаете для проекта? Бюджет низкий, средний или высокий?"
                ],
                "options": ["низкий", "средний", "высокий"],
                "next": "recommendation"
            }
        }
        
        # Варианты переходных фраз
        self.transitions = [
            "Отлично, понял вас. {}",
            "Хорошо, запомнил. {}",
            "Ясно. {}",
            "Интересно! {}",
            "Спасибо за информацию. {}"
        ]
        
        # Варианты уточняющих фраз
        self.clarifications = [
            "Извините, я не совсем понял. Вы имеете в виду {}?",
            "Могли бы вы уточнить? Вы выбираете {}?",
            "Правильно ли я понимаю, что вы выбрали {}?",
            "Хотите выбрать {}? Или что-то другое?",
            "Не уверен, что правильно понял. Вы говорите о {}?"
        ]
        
        # Фразы для неоднозначных ответов
        self.ambiguous_responses = [
            "Я не совсем уверен, что понял ваш выбор. Пожалуйста, выберите один из вариантов: {}.",
            "Хм, не могу точно определить ваш ответ. Пожалуйста, уточните: {}.",
            "Не могу распознать ваш ответ. Можете выбрать из следующих вариантов: {}.",
            "Извините, но мне нужен более конкретный ответ. Варианты: {}.",
            "Давайте уточним. Вы выбираете из: {}."
        ]
        
        # Фразы для завершения диалога
        self.recommendation_intros = [
            "На основе нашего разговора, я подготовил для вас рекомендацию:",
            "Спасибо за ответы! Вот что я могу порекомендовать:",
            "Отлично, теперь у меня есть вся нужная информация. Моя рекомендация:",
            "Основываясь на ваших предпочтениях, я рекомендую:",
            "Проанализировав ваши ответы, я пришел к следующей рекомендации:"
        ]
        
        # Дополнительные комментарии о выборе пользователя
        self.choice_comments = {
            "type": {
                "нативное": [
                    "Нативная разработка - отличный выбор для максимальной производительности.",
                    "Нативные приложения действительно обеспечивают наилучший пользовательский опыт.",
                    "С нативной разработкой вы получите доступ ко всем возможностям платформы."
                ],
                "кроссплатформенное": [
                    "Кроссплатформенная разработка поможет сэкономить ресурсы.",
                    "Разумный выбор! Кроссплатформенные решения становятся всё более мощными.",
                    "Кроссплатформенный подход позволит охватить больше пользователей при меньших затратах."
                ],
                "гибридное": [
                    "Гибридный подход даёт хороший баланс между производительностью и скоростью разработки.",
                    "Гибридные приложения - неплохой компромисс для многих проектов.",
                    "Интересный выбор! Гибридные приложения сочетают преимущества обоих подходов."
                ]
            },
            "platform": {
                "ios": [
                    "iOS - отличная платформа с платежеспособной аудиторией.",
                    "Пользователи iOS часто более активно совершают покупки в приложениях.",
                    "Разработка под iOS обычно более предсказуема из-за ограниченного числа устройств."
                ],
                "android": [
                    "Android даёт доступ к огромной аудитории пользователей.",
                    "С Android вы сможете охватить разнообразные устройства и ценовые сегменты.",
                    "Android предоставляет больше свободы для кастомизации приложения."
                ]
            },
            "experience_level": {
                "новичок": [
                    "Для команды новичков важно выбрать технологии с хорошей документацией и активным сообществом.",
                    "Начинающим разработчикам стоит обратить внимание на инструменты с низким порогом входа.",
                    "Для новичков есть много отличных фреймворков с подробными руководствами."
                ],
                "средний": [
                    "Команда среднего уровня может справиться с большинством популярных технологий.",
                    "С таким уровнем опыта у вас хороший баланс знаний и возможностей для обучения.",
                    "Разработчики среднего уровня могут быстро освоить новые технологии."
                ],
                "эксперт": [
                    "Имея экспертов в команде, вы можете использовать самые продвинутые технологии.",
                    "Эксперты могут эффективно работать с низкоуровневыми инструментами для максимальной оптимизации.",
                    "С опытными разработчиками вы получите максимум от выбранного стека технологий."
                ]
            },
            "performance": {
                "низкая": [
                    "Производительность не является критичной, поэтому можно выбрать более простые технологии.",
                    "При невысоких требованиях к скорости работы можно сосредоточиться на удобстве разработки.",
                    "Проект не требует сложных оптимизаций, что ускорит процесс создания."
                ],
                "средняя": [
                    "Умеренные требования к производительности позволяют выбрать сбалансированные решения.",
                    "Средний уровень оптимизации - хорошее сочетание скорости работы и удобства разработки.",
                    "Можно использовать кроссплатформенные технологии без значительных потерь в производительности."
                ],
                "высокая": [
                    "Для высокой производительности важно выбирать мощные технологии и архитектурные решения.",
                    "Оптимизация кода и серверной части - ключ к высокой скорости работы.",
                    "Ваш проект потребует тщательной работы с памятью, запросами и вычислениями."
                ],
            },
            "speed": {
                "быстрая": [
                    "Если важна скорость, стоит использовать готовые решения и фреймворки.",
                    "Быстрая разработка возможна с низким уровнем кастомизации.",
                    "Можно использовать no-code и low-code платформы для ускорения процесса."
                ],
                "средняя": [
                    "Средняя скорость разработки позволяет достичь хорошего баланса качества и сроков.",
                    "При таком подходе можно детально проработать архитектуру проекта.",
                    "Вы сможете внедрять кастомные функции, не жертвуя слишком много временем."
                ],
                "низкая": [
                    "Длительная разработка подходит для сложных и масштабных проектов.",
                    "Можно сосредоточиться на высоком качестве и продуманности всех деталей.",
                    "Ваш проект получит максимальное внимание к деталям и архитектуре."
                ]
            },
            "cost": {
                "низкий": [
                    "Выбранный бюджет ограничивает технологии, но всё же есть отличные варианты.",
                    "При небольшом бюджете важно оптимизировать затраты и выбирать доступные решения.",
                    "Низкий бюджет не проблема, если грамотно спланировать разработку."
                ],
                "средний": [
                    "С таким бюджетом можно выбрать проверенные технологии без значительных ограничений.",
                    "Средний бюджет позволяет гибко подходить к выбору инструментов и сервисов.",
                    "Вы сможете достичь баланса между ценой и качеством."
                ],
                "высокий": [
                    "С таким бюджетом можно позволить себе лучшие технологии и максимальную оптимизацию.",
                    "Высокий бюджет открывает возможности для использования мощных и современных решений.",
                    "Вы сможете внедрить инновационные подходы и создать качественный продукт."
                ]
            }
        }

    
    def get_random_question(self, state):
        """Получает случайный вопрос для текущего состояния диалога"""
        if state in self.dialog_flow:
            return random.choice(self.dialog_flow[state]["question"])
        return "Что бы вы хотели обсудить дальше?"
    
    def get_next_state(self, current_state, user_choice):
        """Определяет следующее состояние диалога на основе выбора пользователя"""
        if current_state in self.dialog_flow:
            next_state = self.dialog_flow[current_state]["next"]
            if callable(next_state):
                return next_state(user_choice)
            return next_state
        return "recommendation"
    
    def get_options(self, state):
        """Получает возможные варианты ответа для текущего состояния"""
        if state in self.dialog_flow:
            return self.dialog_flow[state]["options"]
        return []
    
    def get_transition_phrase(self, next_question):
        """Создает переходную фразу к следующему вопросу"""
        return random.choice(self.transitions).format(next_question)
    
    def get_clarification_phrase(self, matched_option):
        """Создает уточняющую фразу для сомнительного совпадения"""
        return random.choice(self.clarifications).format(matched_option)
    
    def get_ambiguous_response(self, options):
        """Создает фразу для неоднозначного ответа пользователя"""
        options_str = ", ".join(options)
        return random.choice(self.ambiguous_responses).format(options_str)
    
    def get_recommendation_intro(self):
        """Возвращает вступительную фразу для рекомендации"""
        return random.choice(self.recommendation_intros)
    
    def get_choice_comment(self, category, choice):
        """Возвращает дополнительный комментарий о выборе пользователя"""
        if category in self.choice_comments and choice in self.choice_comments[category]:
            return random.choice(self.choice_comments[category][choice])
        return ""
    
    def process_response(self, current_state, user_message, answers):
        """
        Обрабатывает ответ пользователя и возвращает следующий шаг диалога
        
        Args:
            current_state: Текущее состояние диалога
            user_message: Сообщение пользователя
            answers: Словарь с предыдущими ответами пользователя
            
        Returns:
            tuple: (ответ бота, совпадение ответа)
        """        
        # Для состояния recommendation формируем итоговую рекомендацию
        if current_state == "recommendation":
            recommendation = self.recommendation_engine.get_recommendation(answers)
            return self.get_recommendation_intro() + " " + recommendation, None
        
        # Получаем варианты ответа для текущего состояния
        options = self.get_options(current_state)
        
        # Сопоставляем ответ пользователя с вариантами, используя расширенную функцию
        # Передаем категорию для учета синонимов
        category = current_state if current_state != "start" else "type"
        match, score, needs_clarification = match_user_response(user_message, options, category)
        
        # Если нужно уточнение
        if needs_clarification and match:
            return self.get_clarification_phrase(match), match
        
        # Если нет совпадения
        if not match:
            return self.get_ambiguous_response(options), None
        
        # Сохраняем ответ в зависимости от текущего состояния
        if current_state == "start":
            answers["type"] = match
        elif current_state == "platform":
            answers["platform"] = match
        elif current_state == "experience_level":
            answers["experience_level"] = match
        elif current_state == "performance":
            answers["performance"] = match
        elif current_state == "speed":
            answers["speed"] = match
        elif current_state == "cost":
            answers["cost"] = match
        
        # Определяем следующее состояние
        next_state = self.get_next_state(current_state, match)
        
        # Если следующее состояние - рекомендация, формируем ответ
        if next_state == "recommendation":
            recommendation = self.recommendation_engine.get_recommendation(answers)
            return self.get_recommendation_intro() + " " + recommendation, match
        
        # Иначе формируем следующий вопрос
        next_question = self.get_random_question(next_state)
        
        # Добавляем комментарий о выборе, если он есть
        comment = ""
        if current_state == "start":
            comment = self.get_choice_comment("type", match)
        elif current_state == "platform":
            comment = self.get_choice_comment("platform", match)
        elif current_state == "experience_level":
            comment = self.get_choice_comment("experience_level", match)
        
        if comment:
            comment = " " + comment + " "
        
        response = self.get_transition_phrase(next_question)
        return response.replace("{}", comment + next_question if comment else next_question), match
        
    
# Функция для получения эмбеддинга текста
def get_embedding(text):
    """Получает векторное представление текста с помощью BERT"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Используем среднее значение по всем токенам последнего слоя
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Словарь синонимов для различных категорий
SYNONYMS = {
    "platform": {
        "ios": ["ios", "айфон", "iphone", "айпад", "ipad", "эппл", "apple", "айос", "iphon", "айфоне", "iphone"],
        "android": ["android", "андроид", "андройд", "гугл", "google", "самсунг", "samsung", "андроиде", "pixel"]
    },
    "type": {
        "нативное": ["нативное", "нативная", "нативный", "native", "айфон", "андроид"],
        "кроссплатформенное": ["кроссплатформенное", "кроссплатформенный", "cross-platform", "кросс"],
        "гибридное": ["гибридное", "гибридный", "hybrid", "смешанное"]
    },
    "experience_level": {
        "новичок": ["новичок", "начинающий", "базовый", "новички", "junior", "начинающие", "неопытна"],
        "средний": ["средний", "intermediate", "middle", "опытна"],
        "эксперт": ["эксперт", "продвинутый", "senior", "профессионал", "гений"]
    },
    "performance": {
        "критична": ["критична", "критично", "важна", "высокая", "максимальная"],
        "средняя": ["средняя", "умеренная", "нормальная"],
        "не критична": ["некритичный", "некритична", "неважный", "низкий", "неважно", "неважна"]
    },
    "speed": {
        "быстрая": ["быстрая", "быстро", "срочно", "скорая", "молниеносная"],
        "средняя": ["средняя", "нормальная", "умеренная"],
        "низкая": ["низкая", "медленная", "неспеша", "неторопливая", "долгая", "длительная", "спеша", "небыстрая"]
    },
    "cost": {
        "низкий": ["низкий", "невысокий", "не высокий", "маленький", "ограниченный", "минимальный", "экономный", "небольшой"],
        "средний": ["средний", "умеренный", "нормальный", "немаленький"],
        "высокий": ["высокий", "большой", "значительный", "максимальный"]
    }
}

def normalize_synonyms():
    for category in SYNONYMS:
        print(f"DEBUG: Нормализуем категорию '{category}'")
        for key in SYNONYMS[category]:
            print(f"DEBUG: Ключ '{key}', синонимы до: {SYNONYMS[category][key]}")
            lemmatized = []
            for synonym in SYNONYMS[category][key]:
                if synonym.startswith("не "):  # Если есть "не "
                    base_form = "не" + nlp(synonym[3:])[0].lemma_  # Восстанавливаем "не"
                else:
                    base_form = nlp(synonym)[0].lemma_
                lemmatized.append(base_form)
            SYNONYMS[category][key] = list(set(lemmatized))  # Убираем дубли
            print(f"DEBUG: Ключ '{key}', синонимы после: {SYNONYMS[category][key]}")

# Вызов при старте
normalize_synonyms()

# def preprocess_text(text):
#     """Очищает и нормализует текст перед обработкой BERT."""
    
#     doc = nlp(text.lower().strip())  # Приводим к нижнему регистру и убираем пробелы
#     clean_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]  
#     return " ".join(clean_tokens)

def preprocess_text(text):
    """Очищает и нормализует текст перед обработкой BERT."""
    text = text.lower().strip() # Приводим к нижнему регистру и убираем пробелы
    
    # Добавляем замену для слитных вариантов
    replacements = {
        "не высокий": "низкий",
        "невысокий": "низкий",
        "не низкий": "высокий",
        "ненизкий": "высокий",
        "не быстрой": "низкий",
    }
    for key, value in replacements.items():
        text = text.replace(key, value)

    doc = nlp(text)
    clean_tokens = []
    skip_next = False
    
    for i, token in enumerate(doc):
        if skip_next:
            skip_next = False
            continue
            
        # Объединяем "не" со следующим словом
        if token.text == "не" and i + 1 < len(doc):
            next_token = doc[i + 1]
            merged = f"не{next_token.lemma_}"  # "не" + лемма следующего слова
            clean_tokens.append(merged)
            skip_next = True  # Пропускаем следующее слово, так как уже объединили
        elif not token.is_stop and not token.is_punct:
            clean_tokens.append(token.lemma_)
    
    return " ".join(clean_tokens)

def match_user_response(user_text, options, category=None, threshold=0.65):
    """
    Расширенная функция сопоставления ответа пользователя с возможными вариантами
    с учетом синонимов и категорий
    
    Args:
        user_text: Текст пользователя
        options: Список возможных вариантов
        category: Категория ответа (platform, type, etc.)
        threshold: Порог сходства для определения совпадения
        
    Returns:
        tuple: (лучший вариант, уверенность, флаг уточнения)
    """
    print(f"DEBUG: user_text='{user_text}', category='{category}', options={options}")

    # processed_text = preprocess_text(user_text)
    # print(f"DEBUG: После предобработки: '{processed_text}'")

    user_text = user_text.lower().strip()
    
    # Проверка на прямое совпадение или вхождение
    for option in options:
        print(f"DEBUG: Прямое сравнение '{user_text}' с '{option}', вхождение: {option in user_text}")
        if user_text == option or option in user_text:
            print(f"DEBUG: Прямое совпадение найдено с '{option}'")
            return option, 1.0, False
    
    # Специальная проверка для отрицаний (для слов, начинающихся с "не")
    if user_text.startswith("не") or user_text.startswith("некритич"):
        # Проверяем опции, которые также начинаются с "не"
        for option in options:
            print(f"DEBUG: Прямое сравнение '{user_text}' с '{option}', вхождение: {option in user_text}")
            if option.startswith("не"):
                for synonym in SYNONYMS[category][option]:
                    if user_text == synonym or user_text in synonym.split():
                        print(f"DEBUG: Прямое совпадение найдено с '{option}'")
                        return option, 1.0, False
    
    # Проверка по синонимам если указана категория
    if category and category in SYNONYMS:
        words = user_text.split()
        print(f"DEBUG: Слова после разбивки: {words}")
        
        for option in options:
            print(f"DEBUG: Проверяем опцию '{option}'")
            if option in SYNONYMS[category]:
                print(f"DEBUG: Синонимы для '{option}': {SYNONYMS[category][option]}")
                # Проверяем каждое слово в тексте пользователя на соответствие синонимам
                for word in words:
                    if word in SYNONYMS[category][option]:
                        return option, 1.0, False
                
                # Проверяем на подстроки
                for synonym in SYNONYMS[category][option]:
                    user_words = user_text.split()
                    if synonym in user_words:
                        return option, 1.0, False
    
    # Далее используем исходную логику с BERT для семантического сравнения и простого текстового сравнения
    try:
        user_emb = get_embedding(user_text)
        
        best_match = None
        best_score = -1
        
        for option in options:
            option_emb = get_embedding(option)
            # Косинусное сходство между векторами
            similarity = np.dot(user_emb, option_emb) / (np.linalg.norm(user_emb) * np.linalg.norm(option_emb) + 1e-8)
            
            if similarity > best_score:
                best_score = similarity
                best_match = option
        
        # Определяем, нужно ли уточнение
        needs_clarification = 0.65 <= best_score < 0.8
        
        if best_score >= threshold:
            return best_match, best_score, needs_clarification
        return None, best_score, True
    
    except Exception as e:
        print(f"Ошибка при сопоставлении ответа: {e}")
        # Запасной вариант - простое текстовое сравнение
        best_match = None
        best_score = 0
        
        for option in options:
            # Простое сравнение по словам
            common_words = sum(1 for word in user_text.split() if word in option.split())
            score = common_words / max(len(user_text.split()), len(option.split()), 1)
            
            if score > best_score:
                best_score = score
                best_match = option
        
        if best_score >= 0.3:  # Более низкий порог для запасного метода
            return best_match, best_score, best_score < 0.6
        return None, best_score, True

# class TechRecommendationEngine:
#     def __init__(self, conn, cursor):
#         """
#         Инициализирует движок рекомендаций с подключением к базе данных.
        
#         Args:
#             conn: Соединение с базой данных PostgreSQL
#             cursor: Курсор для выполнения запросов
#         """
#         self.conn = conn
#         self.cursor = cursor
        
#         # Инициализируем таблицу технологий, если она пуста
#         self._initialize_tech_database()
    
#     def _initialize_tech_database(self):
#         """Проверяет, есть ли записи в таблице technologies, и если нет - заполняет её начальными данными"""
#         self.cursor.execute("SELECT COUNT(*) FROM technologies")
#         count = self.cursor.fetchone()[0]
        
#         if count == 0:
#             # Начальные данные для заполнения таблицы technologies
#             tech_data = [
#                 # Нативные решения для iOS
#                 ("Swift", "нативное", "ios", "средний", "критична", "средняя", "средний", 
#                  "Рекомендуем Swift для нативной разработки на iOS. Высокая производительность и поддержка Apple. Компания Airbnb использовала Swift для разработки своего iOS-приложения, что позволило им добиться высокой отзывчивости интерфейса и эффективного использования системных ресурсов. Swift также обеспечивает доступ к полному набору нативных API и функций iOS, что критично для приложений с насыщенным пользовательским интерфейсом."),
                
#                 ("Swift/SwiftUI", "нативное", "ios", "быстрая", "новичок", "средняя", "средний", 
#                  "Рекомендуем SwiftUI для быстрой разработки iOS-приложений начинающими разработчиками. Приложение Calm частично использует SwiftUI для новых модулей. SwiftUI значительно ускоряет создание современных пользовательских интерфейсов благодаря декларативному синтаксису и встроенным компонентам. Визуальный редактор позволяет увидеть изменения в реальном времени без перезапуска приложения, а система расположения элементов автоматически адаптируется к различным размерам экранов и ориентациям."),
                
#                 ("Swift/Objective-C", "нативное", "ios", "высокий", "критична", "средняя", "эксперт", 
#                  "Рекомендуем комбинацию Swift и Objective-C для высокопроизводительных iOS-приложений. Spotify и Slack используют этот подход для своих продуктов. Для экспертов это дает возможность писать новый код на Swift, сохраняя доступ к проверенным временем библиотекам Objective-C. Такой подход обеспечивает максимальную гибкость при разработке и оптимизации критичных компонентов. При достаточном бюджете можно добиться наилучшей производительности и использовать все возможности платформы iOS."),
                
#                 # Нативные решения для Android
#                 ("Kotlin", "нативное", "android", "средний", "критична", "средняя", "средний", 
#                  "Рекомендуем Kotlin для нативной разработки на Android. Опыт Pinterest показал, что переход на Kotlin значительно повысил стабильность приложения и снизил количество сбоев. Kotlin предлагает современный синтаксис, полную совместимость с Java и эффективное управление памятью, что делает его оптимальным выбором для производительных Android-приложений. Google официально поддерживает Kotlin как предпочтительный язык для Android-разработки."),
                
#                 ("Kotlin Native", "нативное", "android", "высокий", "критична", "средняя", "эксперт", 
#                  "Рекомендуем Kotlin с Kotlin Native для высокопроизводительных Android-приложений. Яндекс использует Kotlin во многих своих мобильных приложениях. Kotlin Native позволяет компилировать код непосредственно в нативный машинный код без виртуальной машины Java, что повышает производительность критичных компонентов. Для экспертных команд с высоким бюджетом это дает возможность создавать оптимизированные приложения с потенциальным переиспользованием части логики для других платформ."),
                
#                 ("Jetpack Compose", "нативное", "android", "быстрая", "не критична", "средняя", "новичок", 
#                  "Рекомендуем Jetpack Compose для быстрой разработки Android-приложений новичками. Google Pay и другие приложения Google постепенно переходят на Jetpack Compose. Этот современный инструментарий для создания UI с декларативным подходом значительно упрощает разработку сложных интерфейсов. Живой предпросмотр и горячая перезагрузка ускоряют итерации, а интеграция с остальными библиотеками Jetpack создает целостную экосистему для быстрой разработки качественных приложений."),
                
#                 # Кроссплатформенные решения
#                 ("Flutter", "кроссплатформенное", "любая", "новичок", "средняя", "быстрая", "низкий", 
#                  "Рекомендуем Flutter для кроссплатформенной разработки. Alibaba успешно использовала Flutter для своего приложения Xianyu с аудиторией более 50 млн пользователей. Flutter предлагает богатый набор готовых виджетов, горячую перезагрузку для быстрой итерации и единую кодовую базу для iOS и Android. Dart, используемый во Flutter, имеет пологую кривую обучения, что делает его доступным для новичков при сохранении высокой скорости разработки."),
                
#                 ("React Native", "кроссплатформенное", "любая", "средний", "средняя", "средняя", "средний", 
#                  "Рекомендуем React Native. Instagram и Walmart успешно внедрили React Native в свои мобильные приложения, достигнув баланса между производительностью и скоростью разработки. React Native использует JavaScript и реактивный подход, позволяя разработчикам применять веб-навыки в мобильной разработке. Обеспечивает доступ к нативным компонентам через мосты, что дает лучшую производительность по сравнению с полностью гибридными решениями, сохраняя при этом преимущества единой кодовой базы."),
                
#                 ("Flutter с нативными компонентами", "кроссплатформенное", "любая", "средний", "критична", "средняя", "средний", 
#                  "Рекомендуем Flutter с дополнительными оптимизациями производительности. Alibaba's Xianyu app использует этот подход, достигая высокой производительности при кроссплатформенной разработке. Flutter уже предлагает производительность, близкую к нативной, благодаря компиляции в машинный код. Для критичных сценариев рекомендуется использовать Flutter для UI и бизнес-логики, а для особо требовательных компонентов создавать платформенные каналы к нативному коду. Такой подход обеспечивает хороший баланс между затратами на разработку и итоговой производительностью."),
                
#                 ("NativeScript", "кроссплатформенное", "любая", "эксперт", "критична", "средняя", "высокий", 
#                  "Рекомендуем NativeScript для критичных по производительности кроссплатформенных приложений. Raiffeisen Bank использовал NativeScript для своего мобильного банкинга, получив близкую к нативной производительность. NativeScript обеспечивает прямой доступ к нативным API без JavaScript-мостов, что значительно повышает скорость работы. Для опытных разработчиков NativeScript предлагает мощные инструменты оптимизации и профилирования, а также возможность напрямую использовать нативные библиотеки."),
                
#                 # Гибридные решения
#                 ("Ionic", "гибридное", "любая", "новичок", "не критична", "средняя", "низкий", 
#                  "Рекомендуем Ionic. MarketWatch и Sworkit используют Ionic для своих приложений благодаря его интеграции с Angular, Vue или React. Ionic предлагает широкую библиотеку компонентов пользовательского интерфейса, которые автоматически адаптируются под iOS и Android. Это значительно сокращает время и стоимость разработки, а поддержка веб-технологий (HTML, CSS, JavaScript) делает его доступным для веб-разработчиков без опыта мобильной разработки."),
                
#                 ("Progressive Web App", "гибридное", "любая", "новичок", "не критична", "быстрая", "низкий", 
#                  "Рекомендуем Progressive Web App (PWA) для гибридной разработки. Twitter Lite и Starbucks успешно внедрили PWA, что привело к увеличению вовлеченности пользователей и снижению затрат на разработку. PWA работают в браузере, но предлагают функции, близкие к нативным приложениям, включая офлайн-режим, push-уведомления и доступ к некоторым API устройства. Минимальные требования к установке и автоматические обновления делают PWA идеальным решением для стартапов с ограниченным бюджетом."),
                
#                 ("Quasar Framework", "гибридное", "любая", "средний", "не критична", "быстрая", "средний", 
#                  "Рекомендуем Quasar Framework для гибридных приложений с высокой скоростью разработки. Компания CleverTech использовала Quasar для быстрой разработки корпоративных приложений. Quasar основан на Vue.js и предлагает единую кодовую базу для веб, мобильных и десктопных приложений. Отличается богатой экосистемой готовых компонентов и интеграций, что значительно ускоряет процесс разработки при сохранении качества интерфейса."),
                
#                 ("Ionic с Capacitor", "гибридное", "любая", "средний", "средняя", "средняя", "средний", 
#                  "Рекомендуем Ionic с Capacitor для гибридной разработки. Компания Dow Jones использовала эту комбинацию для своего приложения MarketWatch. Ionic предоставляет богатую библиотеку UI-компонентов и интеграцию с популярными JavaScript-фреймворками, а Capacitor обеспечивает доступ к нативным API устройства. Это сочетание позволяет сократить время и стоимость разработки при сохранении доступа к нативной функциональности. Преимущество этого подхода в единой кодовой базе и возможности использования веб-навыков команды.")
#             ]
            
#             for tech in tech_data:
#                 self.cursor.execute("""
#                     INSERT INTO technologies (name, type, platform, experience_level, performance, speed, cost, recommendation_text)
#                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
#                 """, tech)
            
#             self.conn.commit()

#     def normalize_answer(self, key, value):
#         """Нормализует значения для всех категорий ответов"""
#         mappings = {
#             "type": {
#                 "нативное": "нативное",
#                 "айфон": "нативное",
#                 "натив": "нативное",
#                 "родное": "нативное",
#                 "кроссплатформенное": "кроссплатформенное",
#                 "кроссплатформа": "кроссплатформенное",
#                 "кросс-платформа": "кроссплатформенное",
#                 "гибридное": "гибридное",
#                 "гибрид": "гибридное",
#                 "гибридный": "гибридное"
#             },
#             "platform": {
#                 "ios": "ios",
#                 "айос": "ios",
#                 "apple": "ios",
#                 "айфон": "ios",
#                 "android": "android",
#                 "андроид": "android",
#                 "google": "android"
#             },
#             "experience_level": {
#                 "новички": "новичок",
#                 "новичок": "новичок",
#                 "начинающий": "новичок",
#                 "junior": "новичок",
#                 "средние": "средний",
#                 "средний": "средний",
#                 "мидл": "средний",
#                 "middle": "средний",
#                 "эксперты": "эксперт",
#                 "эксперт": "эксперт",
#                 "senior": "эксперт",
#                 "профи": "эксперт"
#             },
#             "performance": {
#                 "критична": "критична",
#                 "важна": "критична",
#                 "высокая": "критична",
#                 "приоритет": "критична",
#                 "средняя": "средняя",
#                 "умеренная": "средняя",
#                 "некритична": "не критична",
#                 "неважна": "не критична",
#                 "низкая": "не критична",
#                 "некритичный" : "не критична"
#             },
#             "speed": {
#                 "быстрая": "быстрая",
#                 "скорость": "быстрая",
#                 "срочно": "быстрая",
#                 "fast": "быстрая",
#                 "средняя": "средняя",
#                 "умеренная": "средняя",
#                 "medium": "средняя",
#                 "низкая": "низкая",
#                 "медленная": "низкая",
#                 "длительная": "низкая",
#                 "slow": "низкая",
#                 "длительная": "низкая",
#                 "длительна": "низкая"
#             },
#             "cost": {
#                 "низкий": "низкий",
#                 "дешево": "низкий",
#                 "эконом": "низкий",
#                 "low": "низкий",
#                 "средний": "средний",
#                 "medium": "средний",
#                 "обычный": "средний",
#                 "высокий": "высокий",
#                 "дорого": "высокий",
#                 "премиум": "высокий",
#                 "high": "высокий"
#             },
#             "community_support": {
#                 "сильное": "сильное",
#                 "активное": "сильное",
#                 "хорошее": "сильное",
#                 "strong": "сильное",
#                 "среднее": "среднее",
#                 "умеренное": "среднее",
#                 "medium": "среднее",
#                 "слабое": "слабое",
#                 "плохое": "слабое",
#                 "weak": "слабое"
#             }
#         }
        
#         # Приводим значение к нижнему регистру перед проверкой
#         value = value.lower().strip() if isinstance(value, str) else str(value).lower()
        
#         # Специальная обработка для числовых значений бюджета
#         if key == "cost" and value.isdigit():
#             value = int(value)
#             if value < 1000: return "низкий"
#             elif 1000 <= value < 5000: return "средний"
#             else: return "высокий"
        
#         return mappings.get(key, {}).get(value, value)
    
#     def get_recommendation(self, answers):
#         """
#         Гибкий алгоритм подбора рекомендации из БД.
#         Для каждой технологии сравниваем параметры с ответами пользователя и суммируем совпадения.
#         """        
#         if "start" in answers:
#             answers["type"] = answers.pop("start")

#         print(f"DEBUG: Входные данные: {answers}")

#         # Нормализуем ответы пользователя
#         normalized = {}
#         for key in ["type", "platform", "experience_level", "performance", "cost", "speed", "community_support"]:
#             if key in answers and answers[key]:
#                 normalized[key] = self.normalize_answer(key, answers[key].lower().strip())
        
#         print(f"DEBUG: Нормализованные ответы: {normalized}")
        
#         # Определяем, какие параметры указаны пользователем
#         parameters = ["experience_level", "performance", "cost", "speed", "platform", "community_support"]
        
#         # Фильтруем технологии по типу приложения
#         user_type = normalized.get("type")
        
#         # Получаем все технологии из БД по указанному типу
#         cursor.execute("""
#             SELECT id, name, platform, experience_level, performance, cost, speed, recommendation_text 
#             FROM technologies 
#             WHERE type = %s
#         """, (user_type,))
#         technologies = cursor.fetchall()
        
#         best_score = -1
#         best_recommendation = None
        
#         # Для каждой технологии считаем, сколько параметров совпадает
#         for tech in technologies:
#             tech_id, name, platform, exp_level, performance, cost, speed, recommendation = tech
            
#             # Создаем словарь с параметрами технологии
#             tech_params = {
#                 "platform": platform.lower() if platform else "",
#                 "experience_level": exp_level.lower() if exp_level else "",
#                 "performance": performance.lower() if performance else "",
#                 "cost": cost.lower() if cost else "",
#                 "speed": speed.lower() if speed else ""
#             }
            
#             score = 0
#             total = 0
            
#             # Проходим по параметрам, которые указал пользователь
#             for param in parameters:
#                 if param in normalized and param in tech_params:
#                     # Для нативных решений параметр platform должен учитываться отдельно
#                     if param == "platform" and user_type != "нативное":
#                         continue
                        
#                     total += 1
#                     if normalized[param] == tech_params[param]:
#                         score += 1  # Совпадение параметра
#                         score += WEIGHTS.get(param, 1.0)  # Добавляем вес параметра
            
#             # Учитываем фидбэк от пользователей
#             cursor.execute("SELECT SUM(rating) FROM feedback WHERE technology_id = %s", (tech_id,))
#             feedback_result = cursor.fetchone()
#             feedback_score = feedback_result[0] if feedback_result and feedback_result[0] is not None else 0
#             score += feedback_score * 0.5  # Добавляем вес фидбэка
            
#             # Вычисляем долю совпадений для этой технологии
#             ratio = score / total if total > 0 else 0
            
#             print(f"DEBUG: Технология {name}, оценка {score}, доля {ratio}")
            
#             # Если совпадений больше предыдущего и доля больше минимального порога, обновляем лучший выбор
#             if ratio >= 0.5 and score > best_score:
#                 best_score = score
#                 best_recommendation = recommendation

#         if best_recommendation:
#             return best_recommendation

#         return ("На основе ваших ответов сложно дать однозначную рекомендацию. "
#                 "Рассмотрите возможность консультации с экспертом по мобильной разработке "
#                 "для более детального анализа вашего проекта.")

def get_user_session(user_id):
    """Получает текущую сессию пользователя из базы данных"""
    conn.rollback()
    cursor.execute("SELECT state, answers FROM user_sessions WHERE user_id = %s", (user_id,))
    result = cursor.fetchone()
    return {"state": result[0], "answers": json.loads(result[1])} if result else {"state": "start", "answers": {}}

def save_user_session(user_id, state, answers):
    """Сохраняет сессию пользователя в базу данных"""
    answers_json = json.dumps(answers)
    # cursor.execute("REPLACE INTO user_sessions (user_id, state, answers) VALUES (%s, %s, %s)", (user_id, state, answers_json))
    # conn.commit()
    cursor.execute("""
    INSERT INTO user_sessions (user_id, state, answers)
    VALUES (%s, %s, %s)
    ON CONFLICT (user_id) 
    DO UPDATE SET state = EXCLUDED.state, answers = EXCLUDED.answers
    """, (user_id, state, answers_json))
    conn.commit()

def get_next_question(state):
    questions = {
        "start": "Какой тип приложения? (Нативное / Кроссплатформенное / Гибридное)",
        "platform": "Для какой платформы? (iOS / Android)",
        "experience_level": "Уровень опыта команды? (Новичок / Средний / Эксперт)",
        "performance": "Производительность? (Критична / Средняя / Не критична)",
        "speed": "Скорость разработки? (Быстрая / Средняя / Низкая)",
        "cost": "Бюджет? (Низкий / Средний / Высокий)",
    }
    return questions.get(state, "Спасибо! Анализируем данные...")

@app.post("/chat/") 
async def chat_response(user_input: UserMessage):
    """ Обработчик API для чата, который реализует естественный диалог """
    user_id = user_input.user_id
    message = user_input.message.strip()
    
    # Если пользователь написал привет, возвращаем приветственное сообщение с начальным вопросом
    if message.lower() == "привет":
        dialog_manager = NaturalDialogManager()
        # Можно вернуть приветственное сообщение и вопрос для начала диалога
        return {"response": "Привет! Давайте начнем. " + dialog_manager.get_random_question("start")}
    
    # Получаем текущую сессию пользователя
    session = get_user_session(user_id)
    current_state = session["state"]
    answers = session["answers"]
    
    # Инициализируем менеджер диалога
    dialog_manager = NaturalDialogManager()
    
    # Если диалог завершен, предлагаем начать заново
    if current_state == "recommendation":
        if "начать заново" in message.lower() or "новый проект" in message.lower():
            # Сбрасываем сессию и начинаем заново
            save_user_session(user_id, "start", {})
            return {"response": dialog_manager.get_random_question("start")}
        else:
            # Предлагаем дополнительную информацию или начать заново
            return {"response": "Я уже дал вам рекомендацию на основе наших обсуждений. Хотите начать новый проект? Просто скажите 'начать заново'."}
    
    # Обрабатываем текущее состояние диалога
    if current_state in dialog_manager.dialog_flow:
        options = dialog_manager.get_options(current_state)
        
        # Пытаемся сопоставить ответ пользователя с возможными вариантами
        category = current_state if current_state != "start" else "type"
        processed_message = preprocess_text(message)  # Предобрабатываем текст перед анализом
        match, confidence, needs_clarification = match_user_response(processed_message, options, category)

        
        # Если найдено четкое совпадение
        if match and not needs_clarification:
            # Сохраняем ответ
            answers[current_state] = match
            
            # Определяем следующее состояние
            next_state = dialog_manager.get_next_state(current_state, match)
            
            # Добавляем комментарий о выборе пользователя, если он есть
            choice_comment = dialog_manager.get_choice_comment(current_state, match)
            comment_text = f" {choice_comment}" if choice_comment else ""
            
            # Если есть следующее состояние с вопросом
            if next_state in dialog_manager.dialog_flow:
                # Формируем естественный переход к следующему вопросу
                next_question = dialog_manager.get_random_question(next_state)
                save_user_session(user_id, next_state, answers)
                return {"response": f"{comment_text} {dialog_manager.get_transition_phrase(next_question)}"}
            
            # Если диалог завершен, формируем рекомендацию
            elif next_state == "recommendation":
                # Получаем рекомендацию
                recommendation_engine = TechRecommendationEngine(conn, cursor)
                recommendation = recommendation_engine.get_recommendation(answers)
                # print(f"DEBUG: Рекомендация: {recommendation}")
                # print(f"DEBUG: Результат запроса tech_id: {tech_id_result}")
                # print(f"DEBUG:Итоговый tech_id: {tech_id}")

                cursor.execute("""
                    SELECT id FROM technologies 
                    WHERE recommendation_text = %s 
                    LIMIT 1
                """, (recommendation,))
                tech_id_result = cursor.fetchone()
                tech_id = tech_id_result[0] if tech_id_result else None
                
                # Сохраняем рекомендацию и состояние
                save_user_session(user_id, "recommendation", answers)
                
                # Формируем значение platform с учетом ограничений БД
                platform_mapping = {"ios": "iOS", "android": "Android"}
                if answers.get("type") == "нативное":
                    platform_answer = answers.get("platform", "").lower()
                    platform = platform_mapping.get(platform_answer, "iOS")
                elif answers.get("type") == "кроссплатформенное":
                    platform = "Кроссплатформенная"
                elif answers.get("type") == "гибридное":
                    platform = "Гибридная"
                else:
                    platform = "iOS"
                
                try:
                    cursor.execute(
                        """INSERT INTO projects 
                           (user_id, project_name, type, platform, budget, 
                            experience_level, performance, speed, recommendation_text, technology_id) 
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                        (
                            user_id, 
                            f"Проект {user_id}", 
                            answers.get("type", "Неизвестно"), 
                            platform, 
                            answers.get("cost", "средний"), 
                            answers.get("experience_level", "средний"), 
                            answers.get("performance", "средняя"), 
                            answers.get("speed", "средняя"), 
                            recommendation,
                            tech_id
                        )
                    )
                    project_id = cursor.fetchone()[0] # Получаем ID созданного проекта
                    conn.commit()

                    # Записываем в БД recommendation
                    cursor.execute('INSERT INTO recommendations (project_id) VALUES (%s) RETURNING id', (project_id,))
                    recommendation_id = cursor.fetchone()[0]
                    conn.commit()

                    # Записываем в БД recommendation_technologies
                    cursor.execute('INSERT INTO recommended_technologies (recommendation_id, technology_id) VALUES (%s, %s)',
                    (recommendation_id, tech_id))
                    conn.commit()

                except Exception as e:
                    # Логируем ошибку, но продолжаем выполнение
                    print(f"Ошибка при сохранении проекта: {e}")
                
                # Формируем естественный ответ с рекомендацией
                return {
                    "response": f"{comment_text} Анализируя ваш выбор, могу рекомендовать: {recommendation}"
                    + "\n\n\n\nЕсли хотите начать новый проект, просто скажите 'начать заново'."
                }
        
        # Если требуется уточнение
        elif needs_clarification:
            ambiguous_msg = dialog_manager.get_ambiguous_response(options)
            return {"response": ambiguous_msg}
        
        # Если ответ не распознан
        else:
            # Формируем сообщение с просьбой выбрать из списка
            options_text = ", ".join([f"'{opt}'" for opt in options])
            return {
                "response": f"Извините, не совсем понял ваш ответ. Пожалуйста, выберите один из вариантов: {options_text}."
            }
    
    # Если состояние неизвестно
    else:
        # Сбрасываем сессию и начинаем заново
        save_user_session(user_id, "start", {})
        return {"response": f"Давайте начнем сначала. {dialog_manager.get_random_question('start')}"}

class FeedbackRequest(BaseModel):
    user_id: str
    technology_id: int
    rating: int


@app.post("/feedback/")
async def give_feedback(feedback: FeedbackRequest):
    """
    Принимает обратную связь от пользователя.
    rating: 1 = лайк, -1 = дизлайк
    """
    if feedback.rating not in [-1, 1]:
        raise HTTPException(status_code=400, detail="Рейтинг должен быть 1 (лайк) или -1 (дизлайк)")

    try:
        # Начинаем транзакцию
        conn.rollback()  # Очищаем возможные предыдущие состояния
        
        # Проверяем, существует ли технология с указанным ID
        cursor.execute("SELECT 1 FROM technologies WHERE id = %s", (feedback.technology_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Технология с указанным ID не найдена")

        # Вставляем feedback
        cursor.execute(
            "INSERT INTO feedback (user_id, technology_id, rating) VALUES (%s, %s, %s)", 
            (feedback.user_id, feedback.technology_id, feedback.rating)
        )
        conn.commit()

        return {
            "message": "Спасибо за обратную связь!",
            "technology_id": feedback.technology_id,
            "rating": feedback.rating
        }
    
    except HTTPException:
        # Перебрасываем HTTPException как есть
        raise
    
    except Exception as e:
        conn.rollback()
        print(f"Ошибка при сохранении отзыва: {str(e)}")  # Логируем ошибку
        raise HTTPException(
            status_code=500, 
            detail=f"Ошибка сервера при обработке отзыва: {str(e)}"
        )

@app.get("/feedback/{technology_id}")
async def get_technology_feedback(technology_id: int):
    """
    Получает статистику обратной связи для конкретной технологии и рисует график.
    """
    cursor.execute(
        """
        SELECT 
            COUNT(CASE WHEN rating = 1 THEN 1 END) as likes,
            COUNT(CASE WHEN rating = -1 THEN 1 END) as dislikes
        FROM feedback 
        WHERE technology_id = %s
        """, 
        (technology_id,)
    )
    stats = cursor.fetchone()

    if not stats:
        raise HTTPException(status_code=404, detail="Feedback data not found")

    likes = stats[0]
    dislikes = stats[1]

    # Генерация графика
    labels = ['Likes', 'Dislikes']
    values = [likes, dislikes]
    
    fig, ax = plt.subplots()
    ax.bar(labels, values, color=['green', 'red'])
    ax.set_ylabel('Count')
    ax.set_title(f'Feedback Statistics for Technology {technology_id}')

    # Сохраняем график в буфер
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Возвращаем график как изображение
    return StreamingResponse(buf, media_type="image/png")

@app.get("/projects/{user_id}")
def get_user_projects(user_id: str):
    # Сначала выполняем rollback, чтобы очистить возможные проблемы предыдущей транзакции
    conn.rollback()
    
    try:
        cursor.execute("SELECT id, project_name, type, platform, budget, experience_level, recommendation_text, technology_id FROM projects WHERE user_id = %s", (user_id,))
        projects = cursor.fetchall()
        conn.commit()
        
        if not projects:
            return []
            
        return [{"id": p[0], "project_name": p[1], "type": p[2], "platform": p[3], "budget": p[4], "experience_level": p[5],
                  "recommendation_text": "Нет рекомендации" if not p[6] else (p[6][:50] + "...") if len(p[6]) > 50 else p[6], "technology_id": p[7]}
                for p in projects]
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Ошибка при получении проектов: {str(e)}")

@app.delete("/projects/{project_id}")
def delete_project(project_id: int):
    # Начинаем транзакцию
    cursor.execute("SELECT id FROM recommendations WHERE project_id = %s", (project_id,))
    recommendation_ids = cursor.fetchall()

    # Если есть связанные рекомендации
    for recommendation_id in recommendation_ids:
        rec_id = recommendation_id[0]

        # Сначала удаляем записи в recommended_technologies
        cursor.execute("DELETE FROM recommended_technologies WHERE recommendation_id = %s", (rec_id,))

        # Потом удаляем сами рекомендации
        cursor.execute("DELETE FROM recommendations WHERE id = %s", (rec_id,))

    # После этого можно удалить сам проект
    cursor.execute("DELETE FROM projects WHERE id = %s", (project_id,))
    conn.commit()
    return {"message": "Проект успешно удален"}

@app.post("/update_weights/")
def update_weights(new_weights: dict):
    global WEIGHTS
    WEIGHTS.update(new_weights)
    return {"message": "Весовые коэффициенты обновлены", "weights": WEIGHTS}

@app.post("/restart_session/{user_id}")
def restart_session(user_id: str):
    cursor.execute("UPDATE user_sessions SET state = 'start' WHERE user_id = %s", (user_id,))
    conn.commit()
    return {"message": "Форма перезапущена, state сброшен"}

RAMBLER_SMTP_SERVER = "smtp.rambler.ru"
RAMBLER_SMTP_PORT = 465 
RAMBLER_EMAIL = "mikaelnaz@rambler.ru"  
RAMBLER_PASSWORD = "m1razetkatv"  

@app.post("/send_email/")
def send_email(request: EmailRequest):
    user_email = request.user_email
    user_id = request.user_id

    # Получаем рекомендации из БД
    cursor.execute("SELECT recommendation_text FROM projects WHERE user_id = %s", (user_id,))
    recommendations = cursor.fetchall()

    if not recommendations:
        return {"error": "Нет рекомендаций для отправки."}
    
    # 📌 Создаём HTML-шаблон для письма
    message_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Рекомендации</title>
    </head>
    <body style="font-family: Arial, sans-serif; background-color: #f4f4f4; padding: 20px;">
        <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); max-width: 600px; margin: auto;">
            <h2 style="color: #333; text-align: center;">📌 Ваши проекты и рекомендации</h2>
    """

    cursor.execute("""
        SELECT id, project_name, type, platform, budget, experience_level, recommendation_text
        FROM projects WHERE user_id = %s
    """, (user_id,))
    projects = cursor.fetchall()

    if not projects:
        message_content += "<p style='text-align: center; color: red;'>У вас нет проектов.</p>"

    for project in projects:
        project_id, project_name, project_type, platform, budget, experience, recommendation = project
        message_content += f"""
        <div style="background: #f8f9fa; padding: 15px; border-left: 4px solid #007bff; margin-bottom: 10px;">
            <p><strong>📂 Проект:</strong> {project_name}</p>
            <p><strong>💻 Тип:</strong> {project_type}</p>
            <p><strong>📱 Платформа:</strong> {platform}</p>
            <p><strong>💰 Бюджет:</strong> {budget}</p>
            <p><strong>🎓 Опыт команды:</strong> {experience}</p>
        </div>
        <div style="background: #e7f3fe; padding: 15px; border-left: 4px solid #1e88e5; margin-bottom: 10px;">
            <p style="color: #555; font-size: 16px;">💡 <strong>Рекомендация:</strong> {recommendation if recommendation else "Нет рекомендации"}</p>
        </div>
        """

    message_content += """
            <div style="font-size: 14px; color: gray; margin-top: 20px; text-align: center;">
                Mikael's intelligence coorporation © Все права защищены 2025 
            </div>
        </div>
    </body>
    </html>
    """
    
    # # 📌 HTML-письмо
    # message_content = f"""
    # <!DOCTYPE html>
    # <html>
    # <head>
    #     <meta charset="UTF-8">
    #     <title>Рекомендации</title>
    # </head>
    # <body style="font-family: Arial, sans-serif; background-color: #f4f4f4; padding: 20px;">
    #     <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); max-width: 600px; margin: auto;">
    #         <h2 style="color: #333; text-align: center;">📌 Ваши рекомендации по проекту</h2>
    # """

    # for rec in recommendations:
    #     message_content += f"""
    #     <div style="background: #e7f3fe; padding: 15px; border-left: 4px solid #1e88e5; margin-bottom: 10px;">
    #         <p style="color: #555; font-size: 20px;">💡 {rec[0]}</p>
    #     </div>
    #     """

    # message_content += """
    #         <div style="font-size: 16px; color: gray; margin-top: 20px; text-align: center;">
    #             Mikael's intelligence coorporation © Все права защищены 2025 
    #         </div>
    #     </div>
    # </body>
    # </html>
    # """
    # message_content = "Ваши рекомендации:\n\n" + "\n".join([rec[0] for rec in recommendations])

    # Формируем письмо
    msg = MIMEMultipart("alternative")
    msg["From"] = RAMBLER_EMAIL
    msg["To"] = user_email
    msg["Subject"] = "📩 Mikael's intelligence system"
    # msg.attach(MIMEText(message_content, "plain", "utf-8"))

    # 📌 Добавляем HTML-версию письма с принудительным рендерингом
    part_html = MIMEText(message_content, _subtype="html", _charset="utf-8")
    msg.attach(part_html)

    try:
        with smtplib.SMTP_SSL(RAMBLER_SMTP_SERVER, RAMBLER_SMTP_PORT) as server:
            server.login(RAMBLER_EMAIL, RAMBLER_PASSWORD)
            server.sendmail(RAMBLER_EMAIL, user_email, msg.as_string())

        return {"message": "Рекомендации отправлены на email!"}
    except Exception as e:
        return {"error": f"Ошибка при отправке: {str(e)}"}


# Регистрация шрифта с поддержкой кириллицы
font_path = "D:\Диплом\project\DejaVuSans.ttf"
pdfmetrics.registerFont(TTFont("DejaVu", font_path))

@app.get("/export_pdf/{user_id}")
def export_pdf(user_id: str):
    cursor.execute("SELECT project_name, type, platform, budget, experience_level, performance, speed FROM projects WHERE user_id = %s", (user_id,))
    projects = cursor.fetchall()

    if not projects:
        return {"error": "Нет проектов для экспорта."}

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setTitle("Рекомендации по проекту")
    
    # Получаем размеры страницы
    width, height = letter
    # Устанавливаем константы для управления позицией на странице
    top_margin = 750
    bottom_margin = 50
    y_position = top_margin
    page_num = 1  # Начальный номер страницы
    
    def create_new_page():
        nonlocal y_position, page_num
        pdf.setFont("DejaVu", 10)
        pdf.setFillColor(colors.black)
        pdf.drawRightString(width - 50, 30, f"Страница {page_num}")
        pdf.showPage()
        page_num += 1
        y_position = top_margin
        # Добавляем заголовок на новую страницу
        pdf.setFillColor(colors.darkblue)
        pdf.setFont("DejaVu", 16)
        pdf.drawString(100, y_position, "Рекомендации по вашему проекту (продолжение)")
        pdf.line(100, y_position - 5, 500, y_position - 5) # Подчеркивание заголовка
        y_position -= 30
        pdf.setFont("DejaVu", 12)
        pdf.setFillColor(colors.black)

    # Функция для проверки наличия места на странице
    def check_space(needed_space):
        nonlocal y_position
        if y_position - needed_space < bottom_margin:
            create_new_page()
            return True
        return False
    
    # Добавляем заголовок на первую страницу
    pdf.setFillColor(colors.darkblue)
    pdf.setFont("DejaVu", 16)
    pdf.drawString(100, y_position, "Рекомендации по вашему проекту")
    pdf.line(100, y_position - 5, 500, y_position - 5)
    y_position -= 30

    pdf.setFont("DejaVu", 12)
    pdf.setFillColor(colors.black)
    
    for project in projects:
        project_name, project_type, platform, budget, experience, performance, speed = project
        # Проверяем, хватит ли места для нового проекта (примерное значение)
        if check_space(150):
            pass
        
        pdf.setFillColor(colors.darkred)
        pdf.drawString(100, y_position, f"Проект: {project_name}")
        y_position -= 20
        check_space(20)
        
        pdf.setFillColor(colors.black)
        pdf.drawString(100, y_position, f"Тип: {project_type}, Платформа: {platform}")
        y_position -= 20
        # Проверка места
        check_space(20)
        
        pdf.drawString(100, y_position, f"Производительность: {performance}, Скорость разработки: {speed}")
        y_position -= 20
        check_space(20)
        
        pdf.drawString(100, y_position, f"Бюджет: {budget}, Опыт: {experience}")
        y_position -= 20
        check_space(20)
        
        pdf.setFillColor(colors.darkgreen)
        pdf.drawString(100, y_position, "Рекомендация:")
        y_position -= 20
        check_space(20)
        
        pdf.setFillColor(colors.black)
        recommendation_engine = TechRecommendationEngine(conn, cursor)
        words = recommendation_engine.get_recommendation({
            "type": project_type.lower(),
            "experience_level": experience.lower(),
            "cost": budget.lower(),
            "performance": performance.lower(),
            "speed": speed.lower(),
            "platform": platform.lower().replace("iOS", "ios").replace("Android", "android")
              if project_type.lower() == "нативное" else None
        }).split()
        
        line = ""
        for word in words:
            test_line = line + " " + word if line else word
            if pdf.stringWidth(test_line, "DejaVu", 12) < 400:
                line = test_line
            else:
                pdf.drawString(120, y_position, line)
                y_position -= 20
                check_space(20)
                line = word
        if line:
            pdf.drawString(120, y_position, line)
            y_position -= 20
        
        y_position -= 20
        pdf.setFillColor(colors.gray)
        pdf.line(100, y_position + 10, 500, y_position + 10)
        y_position -= 20
        check_space(20)
    
    pdf.setFont("DejaVu", 10)
    pdf.setFillColor(colors.black)
    pdf.drawRightString(width - 50, 30, f"Страница {page_num}")
    pdf.save()
    buffer.seek(0)
    
    headers = {"Content-Disposition": "attachment; filename=recommendations.pdf"}
    return Response(content=buffer.getvalue(), media_type="application/pdf", headers=headers)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

