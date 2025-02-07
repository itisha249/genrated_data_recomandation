import pandas as pd
import streamlit as st

def get_target_users(video_category, video_name, video_id, recommend_score=0):
    # Load datasets
    user_data_path = r"D:\genrated_data_recomadation\genrated_data_recomandation\user_data_main.csv"
    follower_data_path = r"D:\genrated_data_recomadation\genrated_data_recomandation\follower_data.csv"
    video_data_path = r"D:\genrated_data_recomadation\genrated_data_recomandation\video_data_main.csv"

    # Read CSV files
    user_data = pd.read_csv(user_data_path)
    follower_data = pd.read_csv(follower_data_path)
    
    # Select 1000 users most interested in this category
    if video_category in user_data.columns:
        interested_users = user_data[['user_id', video_category]].sort_values(by=video_category, ascending=False).head(1000)
    else:
        interested_users = pd.DataFrame(columns=['user_id'])  # Empty if category not found

    # Select 500 new users with minimal engagement
    engagement_columns = user_data.columns[3:]  # Excluding user_id, country, and language
    user_data['total_engagement'] = user_data[engagement_columns].sum(axis=1)
    new_users = user_data[user_data['total_engagement'] == 0].sample(n=500, random_state=42) if len(user_data[user_data['total_engagement'] == 0]) >= 500 else user_data[user_data['total_engagement'] == 0]

    # Select subscribers (if subscriber data exists)
    if 'subscribed' in user_data.columns:
        subscribers = user_data[user_data['subscribed'] == 1]
    else:
        subscribers = pd.DataFrame(columns=['user_id'])  # Empty if no column found

    # Extract followers of the content creator
    follower_data['following'] = follower_data['following'].astype(str)  # Ensure string format
    followers = follower_data[follower_data['following'].str.contains(str(video_id))][['user_id']]

    # Merge all selected users
    first_group_users = pd.concat([followers, interested_users[['user_id']], new_users[['user_id']], subscribers[['user_id']]]).drop_duplicates()
    
    st.write("### First Group of Users Targeted:")
    st.dataframe(first_group_users)
    
    # Ask user if the video is temporarily viral
    viral_input = st.radio("Do you want to simulate the video as viral?", ('No', 'Yes'))
    if viral_input == 'Yes':
        recommend_score_updated = 80  # Simulating viral video
    else:
        st.write("Waiting for engagement data...")
        
        # Reload the video data to get the updated recommendation score
        video_data = pd.read_csv(video_data_path)
        video_entry = video_data[video_data['id'] == video_id]
        
        if not video_entry.empty:
            recommend_score_updated = video_entry['recommend_score'].values[0] * 100  # Convert to percentage if necessary
        else:
            st.write("Video data not found after 2 days. Cannot check engagement.")
            return
    
    if recommend_score_updated >= 80:
        st.write("### The video has reached 80% engagement, expanding audience...")
        additional_new_users = user_data.sample(n=1000, random_state=42)  # Select 1000 random users
        extended_users = pd.concat([first_group_users, additional_new_users[['user_id']]]).drop_duplicates()
        st.write("### Extended Audience Targeted:")
        st.dataframe(extended_users)
    else:
        st.write("### The video did not reach 80% engagement. No further sharing.")

# Streamlit UI
def main():
    st.title("Viral Video Targeting System")
    video_category = st.text_input("Enter Video Category:")
    video_name = st.text_input("Enter Video Name:")
    video_id = st.number_input("Enter Video ID:", min_value=1, step=1)
    
    if st.button("Run Targeting System"):
        get_target_users(video_category, video_name, video_id)

if __name__ == "__main__":
    main()
