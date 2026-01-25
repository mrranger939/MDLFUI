import torch
import os
import random
import json
import numpy as np
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from data_utils import clean_tweet_v2, get_behavioral_vector, prepare_image

def get_voting_samples(user_path, num_samples=5):
    """Fetches up to 5 random samples for robust voting."""
    timeline_path = os.path.join(user_path, "timeline.txt")
    samples = []
    
    # 1. Gather all possible tweets
    if os.path.exists(timeline_path):
        try:
            with open(timeline_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                random.shuffle(lines)
                for line in lines:
                    if len(samples) >= num_samples: break
                    try:
                        data = json.loads(line)
                        if data.get("text"):
                            samples.append(data["text"])
                    except: continue
        except: pass
    
    # 2. Get Random Image (Shared across samples for efficiency)
    try:
        images = [f for f in os.listdir(user_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        image_path = os.path.join(user_path, random.choice(images)) if images else None
    except: image_path = None
    
    return samples, image_path

def evaluate_whole_dataset(model, tokenizer, bert, vis_extractor, device, dataset_root="data/MultiModalDataset"):
    y_true = []
    y_pred = []
    
    categories = {"positive": 1.0, "negative": 0.0}
    
    st.write("Initializing Voting Evaluation (5 samples per user)...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Calculate Total
    total_users = sum([len(os.listdir(os.path.join(dataset_root, cat))) for cat in categories if os.path.exists(os.path.join(dataset_root, cat))])
    processed = 0
    
    for category, label in categories.items():
        cat_path = os.path.join(dataset_root, category)
        if not os.path.exists(cat_path): continue
        
        users = [d for d in os.listdir(cat_path) if os.path.isdir(os.path.join(cat_path, d))]
        
        for user_id in users:
            user_path = os.path.join(cat_path, user_id)
            
            # 1. Get Samples for Voting
            tweets, img_path = get_voting_samples(user_path, num_samples=5)
            if not tweets: 
                processed += 1
                continue
                
            # 2. Process Image Once (Shared)
            v_vec = None
            if img_path:
                img_tensor = prepare_image(img_path, device)
                if img_tensor is not None:
                    with torch.no_grad():
                        v_vec = torch.flatten(vis_extractor(img_tensor), 1)
            if v_vec is None: v_vec = torch.zeros((1, 1280)).to(device)
            
            # 3. Process Behavior Once (Shared)
            b_vec = get_behavioral_vector(user_path).to(device)
            
            # 4. Run Model on 5 Samples & Average
            user_probs = []
            for text in tweets:
                cleaned = clean_tweet_v2(text)
                t_in = tokenizer(cleaned, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
                with torch.no_grad():
                    t_vec = bert(**t_in).last_hidden_state[:, 0, :]
                    prob = model(t_vec, v_vec, b_vec).item()
                    user_probs.append(prob)
            
            # 5. VOTING LOGIC
            avg_prob = np.mean(user_probs)
            y_true.append(label)
            y_pred.append(1.0 if avg_prob > 0.5 else 0.0)
            
            processed += 1
            if total_users > 0:
                progress_bar.progress(min(processed / total_users, 1.0))
                status_text.text(f"Processed {processed}/{total_users} users")

    status_text.text("Evaluation Complete!")
    progress_bar.empty()
    
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "Confusion Matrix": confusion_matrix(y_true, y_pred)
    }
    return metrics