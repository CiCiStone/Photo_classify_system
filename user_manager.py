import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import os
import json

def create_connection():
    conn = sqlite3.connect('users.db')
    return conn

def create_table():
    conn = create_connection()
    conn.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    ''')
    conn.close()

def register_user(username, password):
    conn = create_connection()
    hashed_password = generate_password_hash(password)
    conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
    conn.commit()
    conn.close()

def validate_user(username, password):
    conn = create_connection()
    user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    conn.close()
    if user and check_password_hash(user[2], password):
        return True
    return False

create_table()

f = json.load(open("./discription.json", 'r', encoding='utf-8'))

class_dict = f[0]
class_descriptions = f[1]

