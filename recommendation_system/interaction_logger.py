import time
from database import get_db_connection

# Function to log user interactions
def log_user_interaction(user_id, video_id, action):
    """
    Logs user interactions (click, watch, skip) in SQLite.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = time.time()
    
    cursor.execute("INSERT INTO interactions (user_id, video_id, action, timestamp) VALUES (?, ?, ?, ?)",
                   (user_id, video_id, action, timestamp))
    
    conn.commit()
    conn.close()
    
    print(f"✅ Interaction logged: User {user_id}, Video {video_id}, Action {action}")
