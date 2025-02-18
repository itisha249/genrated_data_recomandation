from database import initialize_db
from interaction_logger import log_user_interaction
from adaptive_ranking import get_adaptive_recommendations

# Initialize Database
initialize_db()

# Log some example interactions
log_user_interaction(user_id=101, video_id=915, action="click")
log_user_interaction(user_id=101, video_id=831, action="watch")
log_user_interaction(user_id=102, video_id=947, action="skip")

# Retrieve and display ranked recommendations
adaptive_recommendations = get_adaptive_recommendations()
print("🔹 Adaptive Ranked Recommendations:")
print(adaptive_recommendations)
