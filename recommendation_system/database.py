import sqlite3

# Database connection function
def get_db_connection():
    db_path = "user_interactions.db"
    conn = sqlite3.connect(db_path)
    return conn

# Create the interactions table if it doesn't exist
def initialize_db():
    conn = get_db_connection()
    cursor = conn.cursor()
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
    conn.close()
