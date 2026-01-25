import re
import os
import json
import demoji
import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
FPS_SINGULAR = {"i", "me", "my", "mine", "i'm", "i’ve", "i'll", "i’d"}
FPS_PLURAL = {"we", "us", "our", "ours", "we're", "we’ve", "we'll", "we’d"}

def clean_tweet_v2(text):
    if not text: return ""
    text = demoji.replace_with_desc(text, sep=" ")
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'&\w+;', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def get_behavioral_vector(user_folder_path):
    timeline_path = os.path.join(user_folder_path, "timeline.txt")
    if not os.path.exists(timeline_path): return torch.zeros((1, 9))

    all_valid_tweets = []
    all_timestamps = []
    
    try:
        image_ids = {os.path.splitext(f)[0] for f in os.listdir(user_folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))}
    except:
        image_ids = set()

    with open(timeline_path, "r", encoding='utf-8') as f:
        for line in f:
            try:
                tweet = json.loads(line)
                text = tweet.get("text", "")
                tweet_id = str(tweet.get("id_str", tweet.get("id")))
                created_at = tweet.get("created_at")
                if not text: continue
                
                tweet_data = {"text": text, "id": tweet_id, "clean": clean_tweet_v2(text)}
                all_valid_tweets.append(tweet_data)
                
                if created_at:
                    all_timestamps.append(datetime.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y'))
            except: continue

    if not all_valid_tweets: return torch.zeros((1, 9))

    # --- PART A: TEMPORAL (All History) ---
    hours = [d.hour for d in all_timestamps]
    unique_days = len(set(d.date() for d in all_timestamps))
    
    late_night = sum(1 for h in hours if 0 <= h <= 5) / len(hours) if hours else 0.0
    freq = len(all_valid_tweets) / unique_days if unique_days > 0 else 1.0
    time_var = np.std(hours) if hours else 5.0

    # --- PART B: PSYCHO (Last 50 Only) ---
    recent_tweets = all_valid_tweets[-50:] 
    
    sentiments, total_words, fps_count, fpp_count = [], 0, 0, 0
    mention_count, unique_mentions, media_count = 0, set(), 0
    
    for t in recent_tweets:
        tokens = t["clean"].split()
        total_words += len(tokens)
        fps_count += sum(1 for tok in tokens if tok in FPS_SINGULAR)
        fpp_count += sum(1 for tok in tokens if tok in FPS_PLURAL)
        sentiments.append(analyzer.polarity_scores(t["clean"])["compound"])
        
        mentions = re.findall(r'@\w+', t["text"])
        mention_count += len(mentions)
        unique_mentions.update(mentions)
        
        if t["id"] in image_ids: media_count += 1

    tweet_count_50 = len(recent_tweets) if len(recent_tweets) > 0 else 1
    if total_words == 0: total_words = 1

    # --- ASSEMBLE VECTOR ---
    # NEW: We apply np.log1p to "unique_mentions" (Index 8) to normalize the scale
    vector = [
        late_night, freq, time_var,
        fps_count / total_words, fpp_count / total_words,
        np.std(sentiments) if len(sentiments) > 1 else 0.0,
        media_count / tweet_count_50,
        mention_count / tweet_count_50,
        np.log1p(len(unique_mentions)) # <--- FIXED: Log Transform helps scaling!
    ]
    return torch.tensor([vector], dtype=torch.float32)

def prepare_image(image_input, device):
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if isinstance(image_input, str):
            if not os.path.exists(image_input): return None
            img = Image.open(image_input).convert('RGB')
        else:
            img = Image.open(image_input).convert('RGB')
        return transform(img).unsqueeze(0).to(device)
    except: return None