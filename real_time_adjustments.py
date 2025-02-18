import sqlite3
import json
import time
import pandas as pd

# Connect to SQLite database
db_path = "user_interactions.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create interactions table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        video_id INTEGER,
        action TEXT,
        timestamp REAL
    )
''')
conn.commit()

# Function to log user interactions
def log_user_interaction(user_id, video_id, action):
    """
    Logs user interactions (click, watch, skip) in SQLite.
    """
    timestamp = time.time()
    cursor.execute("INSERT INTO interactions (user_id, video_id, action, timestamp) VALUES (?, ?, ?, ?)",
                   (user_id, video_id, action, timestamp))
    conn.commit()
    print(f"✅ Interaction logged: User {user_id}, Video {video_id}, Action {action}")

# Function to retrieve interaction data
def get_interaction_data():
    """
    Retrieves and processes interaction data from SQLite.
    """
    df = pd.read_sql_query("SELECT * FROM interactions", conn)
    return df

# Example Usage
log_user_interaction(user_id=101, video_id=915, action="click")
log_user_interaction(user_id=101, video_id=831, action="watch")
log_user_interaction(user_id=102, video_id=947, action="skip")

# Retrieve logged interactions
df = get_interaction_data()
print("🔹 User Interaction Data:")
print(df)
