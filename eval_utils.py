import torch
import os
import random
import json
import numpy as np
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from data_utils import clean_tweet_v2, get_behavioral_vector, prepare_image

def get_single_sample(user_path, seed=None):
    """
    Get ONE random tweet and ONE random image.
    If seed is provided, makes selection deterministic.
    """
    if seed is not None:
        random.seed(seed)
    
    timeline_path = os.path.join(user_path, "timeline.txt")
    
    # 1. Get ONE random tweet (with retry logic)
    tweet_text = None
    if os.path.exists(timeline_path):
        try:
            with open(timeline_path, 'r', encoding='utf-8') as f:
                lines = [l for l in f.readlines() if l.strip()]
                
                # Try up to 5 times to get a valid tweet
                for attempt in range(min(5, len(lines))):
                    if lines:
                        random_line = random.choice(lines)
                        try:
                            data = json.loads(random_line)
                            text = data.get("text", "")
                            if text:  # Found valid text
                                tweet_text = text
                                break
                        except:
                            continue
        except:
            pass
    
    # 2. Get ONE random image
    image_path = None
    try:
        images = [f for f in os.listdir(user_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if images:
            image_path = os.path.join(user_path, random.choice(images))
    except:
        pass
    
    return tweet_text, image_path

def evaluate_whole_dataset(model, tokenizer, bert, vis_extractor, device, 
                          dataset_root="data/MultiModalDataset", 
                          deterministic_seed=42):
    """
    Evaluate entire dataset with improved stability.
    
    Args:
        deterministic_seed: If provided, uses same random selections each run
    """
    y_true = []
    y_pred = []
    
    categories = {"positive": 1.0, "negative": 0.0}
    
    st.write("Evaluating (Single Sample Per User - Matches Notebook)...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Statistics tracking
    skipped_count = 0
    processed_count = 0
    
    # Calculate Total
    total_users = sum([len(os.listdir(os.path.join(dataset_root, cat))) 
                      for cat in categories if os.path.exists(os.path.join(dataset_root, cat))])
    
    current_idx = 0
    
    for category, label in categories.items():
        cat_path = os.path.join(dataset_root, category)
        if not os.path.exists(cat_path): 
            continue
        
        users = sorted([d for d in os.listdir(cat_path) if os.path.isdir(os.path.join(cat_path, d))])
        
        for user_id in users:
            user_path = os.path.join(cat_path, user_id)
            
            # Use deterministic seed if provided (hash user_id for consistency)
            user_seed = hash(user_id) if deterministic_seed else None
            
            # Get single sample with retry logic
            tweet_text, img_path = get_single_sample(user_path, seed=user_seed)
            
            if not tweet_text:
                skipped_count += 1
                current_idx += 1
                continue
            
            # Process Text
            cleaned = clean_tweet_v2(tweet_text)
            t_in = tokenizer(cleaned, return_tensors="pt", padding=True, 
                           truncation=True, max_length=128).to(device)
            with torch.no_grad():
                t_vec = bert(**t_in).last_hidden_state[:, 0, :]
            
            # Process Image
            v_vec = None
            if img_path:
                img_tensor = prepare_image(img_path, device)
                if img_tensor is not None:
                    with torch.no_grad():
                        v_vec = torch.flatten(vis_extractor(img_tensor), 1)
            if v_vec is None:
                v_vec = torch.zeros((1, 1280)).to(device)
            
            # Process Behavior
            b_vec = get_behavioral_vector(user_path).to(device)
            
            # Debug: Print first user's behavioral vector
            if processed_count == 0:
                st.write(f"**Debug - First user behavioral vector:** {b_vec.cpu().numpy().flatten()[:9]}")
            
            # Predict
            with torch.no_grad():
                prob = model(t_vec, v_vec, b_vec).item()
            
            y_true.append(label)
            y_pred.append(1.0 if prob > 0.5 else 0.0)
            
            processed_count += 1
            current_idx += 1
            
            if total_users > 0:
                progress_bar.progress(min(current_idx / total_users, 1.0))
                status_text.text(f"Processed {current_idx}/{total_users} users ({skipped_count} skipped)")

    status_text.text(f"Evaluation Complete! Processed {processed_count} users, skipped {skipped_count}")
    progress_bar.empty()
    
    # Display statistics
    st.info(f"""
    **Evaluation Statistics:**
    - Total users: {total_users}
    - Successfully evaluated: {processed_count}
    - Skipped (no valid tweets): {skipped_count}
    - Skip rate: {skipped_count/total_users*100:.1f}%
    """)
    
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "Confusion Matrix": confusion_matrix(y_true, y_pred)
    }
    return metrics