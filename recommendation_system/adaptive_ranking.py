from reinforcement_learning import get_interaction_data, thompson_sampling_ranking

# Function to get ranked recommendations based on real-time data
def get_adaptive_recommendations():
    """
    Fetch and rank recommended videos dynamically using Thompson Sampling.
    """
    df = get_interaction_data()
    ranked_videos = thompson_sampling_ranking(df)
    return [video[0] for video in ranked_videos]
