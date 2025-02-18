import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import joblib

# Load video dataset
VIDEO_DATA_PATH = r"D:\genrated_data_recomadation\genrated_data_recomandation\video_data_main.csv"
video_data = pd.read_csv(VIDEO_DATA_PATH, dtype=str, low_memory=False)

# Load pre-trained BERT model & tokenizer
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Ensure required columns exist
if "title" not in video_data.columns:
    raise KeyError("❌ The 'title' column is missing in the dataset!")

# Use title only since 'description' is missing
video_data["combined_text"] = video_data["title"]

# Function to generate BERT embeddings
def get_embedding(text, model, tokenizer):
    """Generate sentence embeddings using BERT."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # CLS token representation

# Compute embeddings for each video
video_data["embedding"] = video_data["combined_text"].apply(lambda x: get_embedding(str(x), model, tokenizer))

# Save embeddings for future use
joblib.dump(video_data, r"D:\genrated_data_recomadation\genrated_data_recomandation\video_import_emmbeding.pkl")

print("✅ Video embeddings generated & saved!")
