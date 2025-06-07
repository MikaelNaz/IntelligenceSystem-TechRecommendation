# import sqlite3

# conn = sqlite3.connect("tech_stack.db")
# cursor = conn.cursor()

import psycopg2

conn = psycopg2.connect(
    dbname="tech_stack",
    user="postgres", 
    password="3932323",  
    host="localhost",
    port="5432"
)
cursor = conn.cursor()
print('Подключение успешно')

cursor.execute('''
CREATE TABLE IF NOT EXISTS technologies (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    platform TEXT NOT NULL,
    experience_level TEXT NOT NULL,
    performance TEXT NOT NULL,
    speed TEXT NOT NULL,
    cost TEXT NOT NULL,
    recommendation_text TEXT NOT NULL
)
''')
conn.commit()

cursor.execute("""
CREATE TABLE feedback_new (
    id SERIAL PRIMARY KEY,
    user_id TEXT,
    technology_id INTEGER REFERENCES technologies(id),
    rating INTEGER
)
""")
conn.commit()

# Создание таблицы пользователей
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    user_id TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')
conn.commit()

# Создание таблицы сессий пользователей
cursor.execute('''
CREATE TABLE IF NOT EXISTS user_sessions (
    user_id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    answers TEXT NOT NULL
)
''')
conn.commit()

# Создание новой таблицы projects, где столбец platform допускает NULL
cursor.execute('''
CREATE TABLE IF NOT EXISTS projects (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_name TEXT NOT NULL,
    type TEXT NOT NULL,
    platform TEXT NOT NULL,
    budget TEXT NOT NULL,
    experience_level TEXT NOT NULL,
    performance TEXT NOT NULL,
    speed TEXT NOT NULL,
    recommendation_text TEXT NOT NULL,
    technology_id INTEGER REFERENCES technologies(id)
)
''')

# Создание таблицы рекомендаций
cursor.execute('''
CREATE TABLE IF NOT EXISTS recommendations (
    id SERIAL PRIMARY KEY,
    project_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects(id)
)
''')
conn.commit()

# Таблица для связи рекомендаций и технологий (многие-ко-многим)
cursor.execute('''
CREATE TABLE IF NOT EXISTS recommended_technologies (
    recommendation_id INTEGER NOT NULL,
    technology_id INTEGER NOT NULL,
    FOREIGN KEY (recommendation_id) REFERENCES recommendations(id),
    FOREIGN KEY (technology_id) REFERENCES technologies(id),
    PRIMARY KEY (recommendation_id, technology_id)
)
''')
conn.commit()

# Создание таблицы правил
cursor.execute('''
CREATE TABLE IF NOT EXISTS rules (
    id SERIAL PRIMARY KEY,
    condition TEXT NOT NULL,
    technology_id INTEGER NOT NULL,
    FOREIGN KEY (technology_id) REFERENCES technologies(id)
)
''')
conn.commit()
