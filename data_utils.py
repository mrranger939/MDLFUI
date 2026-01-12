import re
import demoji
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def clean_tweet_v2(text):
    if not text: return ""
    text = demoji.replace_with_desc(text, sep=" ")
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'&\w+;', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def extract_single_behavioral(text):
    tokens = text.split()
    fps = {"i", "me", "my", "mine", "i'm", "i’ve", "i'll", "i’d"}
    fpp = {"we", "us", "our", "ours", "we're", "we’ve", "we'll", "we’d"}
    
    fps_count = sum(1 for t in tokens if t in fps) / len(tokens) if tokens else 0
    fpp_count = sum(1 for t in tokens if t in fpp) / len(tokens) if tokens else 0
    sentiment = analyzer.polarity_scores(text)["compound"]
    
    # We return 9 features total to match model input (placeholders for the rest)
    return torch.tensor([[fps_count, fpp_count, sentiment, 0.1, 0.2, 0.0, 0.5, 0.3, 2.0]], dtype=torch.float32)

def prepare_image(image, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if isinstance(image, str): image = Image.open(image).convert('RGB')
    return transform(image).unsqueeze(0).to(device)