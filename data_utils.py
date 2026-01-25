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

    sentiments, total_words, fps_count, fpp_count = [], 0, 0, 0
    mention_count, unique_mentions, tweet_count, media_count = 0, set(), 0, 0
    timestamps = []
    
    image_ids = {os.path.splitext(f)[0] for f in os.listdir(user_folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))}

    with open(timeline_path, "r", encoding='utf-8') as f:
        for line in f:
            try:
                tweet = json.loads(line)
                text = tweet.get("text", "")
                tweet_id = str(tweet.get("id_str", tweet.get("id")))
                created_at = tweet.get("created_at")
                if not text: continue
                
                cleaned = clean_tweet_v2(text)
                tokens = cleaned.split()
                total_words += len(tokens)
                fps_count += sum(1 for t in tokens if t in FPS_SINGULAR)
                fpp_count += sum(1 for t in tokens if t in FPS_PLURAL)
                sentiments.append(analyzer.polarity_scores(cleaned)["compound"])
                
                mentions = re.findall(r'@\w+', text)
                mention_count += len(mentions)
                unique_mentions.update(mentions)
                if tweet_id in image_ids: media_count += 1
                
                if created_at:
                    timestamps.append(datetime.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y'))
                tweet_count += 1
            except: continue

    if tweet_count == 0: return torch.zeros((1, 9))

    hours = [d.hour for d in timestamps]
    unique_days = len(set(d.date() for d in timestamps))
    
    # The Exact 9 Features from Notebook Cell 10
    vector = [
        sum(1 for h in hours if 0 <= h <= 5) / len(hours) if hours else 0, # LateNight
        tweet_count / unique_days if unique_days > 0 else 1.0,           # Freq
        np.std(hours) if hours else 5.0,                                # Var
        fps_count / total_words if total_words > 0 else 0,              # SelfFocus
        fpp_count / total_words if total_words > 0 else 0,              # Collective
        np.std(sentiments) if len(sentiments) > 1 else 0.0,             # Volatility
        media_count / tweet_count,                                      # MediaRatio
        mention_count / tweet_count,                                    # Mentions
        len(unique_mentions)                                            # Circle
    ]
    return torch.tensor([vector], dtype=torch.float32)

def prepare_image(image_input, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_input).convert('RGB')
    return transform(img).unsqueeze(0).to(device)