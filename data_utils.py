import re
import os
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

def get_behavioral_vector(user_path=None):
    """
    Simulates the 9-dimensional behavioral vector from your ipynb.
    Indexes: 0-2 (Temporal), 3-8 (Psycholinguistic)
    """
    # In a production scenario, you would calculate these using the logic in cell 10
    # For the UI, we provide standard weights if a user folder isn't fully processed
    return torch.tensor([[0.2, 5.0, 2.5, 0.05, 0.02, 0.1, 0.2, 0.5, 10.0]], dtype=torch.float32)

def prepare_image(uploaded_file, device):
    """Handles the Streamlit UploadedFile object correctly"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # IMPORTANT: Open the file buffer as a PIL Image first
    img = Image.open(uploaded_file).convert('RGB')
    return transform(img).unsqueeze(0).to(device)