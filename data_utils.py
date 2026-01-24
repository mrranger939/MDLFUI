import re
import os
import json
import demoji
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# EXACT sets from Cell 7 of your notebook
FIRST_PERSON_SINGULAR = {"i", "me", "my", "mine", "i'm", "i’ve", "i'll", "i’d"}
FIRST_PERSON_PLURAL = {"we", "us", "our", "ours", "we're", "we’ve", "we'll", "we’d"}

def clean_tweet_v2(text):
    if not text: return ""
    text = demoji.replace_with_desc(text, sep=" ")
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'&\w+;', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def get_behavioral_vector(user_folder_path):
    """
    Implements the EXACT extraction logic from Cells 7 & 10 of FinalProject.ipynb
    """
    timeline_path = os.path.join(user_folder_path, "timeline.txt")
    if not os.path.exists(timeline_path):
        return torch.zeros((1, 9)) # Fallback

    sentiments, total_words, fps_count, fpp_count = [], 0, 0, 0
    mention_count, unique_mentions, tweet_count, media_count = 0, set(), 0, 0
    timestamps = []
    
    # Track media using file existence (Cell 7 logic)
    image_ids = {os.path.splitext(f)[0] for f in os.listdir(user_folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}

    with open(timeline_path, "r", encoding='utf-8') as f:
        for line in f:
            try:
                tweet = json.loads(line) # MUST parse as JSON
                text = tweet.get("text", "")
                tweet_id = str(tweet.get("id_str", tweet.get("id")))
                created_at = tweet.get("created_at")

                if not text: continue
                
                # Preprocessing matches training
                cleaned = clean_tweet_v2(text)
                tokens = cleaned.split()
                total_words += len(tokens)
                
                # Psycholinguistic counts
                fps_count += sum(1 for t in tokens if t in FIRST_PERSON_SINGULAR)
                fpp_count += sum(1 for t in tokens if t in FIRST_PERSON_PLURAL)
                sentiments.append(analyzer.polarity_scores(cleaned)["compound"])
                
                # Social metrics
                mentions = re.findall(r'@\w+', text)
                mention_count += len(mentions)
                unique_mentions.update(mentions)
                
                if tweet_id in image_ids: media_count += 1
                
                # Temporal metrics
                if created_at:
                    dt = datetime.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y')
                    timestamps.append(dt)
                
                tweet_count += 1
            except: continue

    if tweet_count == 0 or total_words == 0: return torch.zeros((1, 9))

    # Calculate final 9 features
    hours = [d.hour for d in timestamps]
    late_night = sum(1 for h in hours if 0 <= h <= 5) / len(hours) if hours else 0.1
    unique_days = len(set(d.date() for d in timestamps))
    posts_day = tweet_count / unique_days if unique_days > 0 else 1.0
    time_var = np.std(hours) if hours else 5.0

    vector = [
        late_night, posts_day, time_var,                # Indices 0-2 (Temporal)
        fps_count / total_words, fpp_count / total_words, # Indices 3-4 (Focus)
        np.std(sentiments) if len(sentiments) > 1 else 0.0, # Index 5 (Volatility)
        media_count / tweet_count,                      # Index 6 (Media)
        mention_count / tweet_count,                    # Index 7 (Mentions)
        len(unique_mentions)                            # Index 8 (Circle)
    ]
    return torch.tensor([vector], dtype=torch.float32)

def prepare_image(uploaded_file, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(uploaded_file).convert('RGB')
    return transform(img).unsqueeze(0).to(device)