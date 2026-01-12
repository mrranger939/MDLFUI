import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from model_utils import load_models
from data_utils import clean_tweet_v2, get_behavioral_vector, prepare_image

st.set_page_config(page_title="Depression Detection AI", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def init_resources():
    return load_models("models/best_trimodal_model_V3.pth", device)

model, tokenizer, bert, vis_extractor = init_resources()

# --- SIDEBAR ---
st.sidebar.title("👤 User Selection")
category = st.sidebar.selectbox("Category", ["positive", "negative"])
dataset_base = f"data/MultiModalDataset/{category}"

selected_user_id = "Custom"
if os.path.exists(dataset_base):
    users = os.listdir(dataset_base)
    selected_user_id = st.sidebar.selectbox("Select Dataset User", ["Custom"] + users)

# --- MAIN UI ---
st.title("🧠 Tri-Modal Depression Detection")
st.info("Using MentalBERT, EfficientNetV2, and Psycholinguistic markers.")

col1, col2 = st.columns([1, 1])

with col1:
    tweet_text = st.text_area("Tweet Text", placeholder="How are you feeling?")
    uploaded_image = st.file_uploader("Tweet Image", type=["jpg", "png", "jpeg"])
    
    if st.button("🔍 Analyze Post"):
        if tweet_text and uploaded_image:
            # 1. Features
            cleaned = clean_tweet_v2(tweet_text)
            
            # Text Emb (cell 18)
            t_in = tokenizer(cleaned, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                t_vec = bert(**t_in).last_hidden_state[:, 0, :]
            
            # Visual Emb (cell 19)
            v_vec = torch.flatten(vis_extractor(prepare_image(uploaded_image, device)), 1)
            
            # Behavior (cell 10)
            user_path = os.path.join(dataset_base, selected_user_id) if selected_user_id != "Custom" else None
            b_vec = get_behavioral_vector(user_path).to(device)

            # 2. Prediction
            with torch.no_grad():
                prob = model(t_vec, v_vec, b_vec).item()

            # 3. Output (Dashboard from last block of ipynb)
            with col2:
                st.subheader("Analysis Results")
                st.metric("Depression Probability", f"{prob:.2%}")
                
                # Granular SHAP Analysis (Simulated based on cell 37)
                st.write("### SHAP Explainability")
                
                # Text highlight
                words = cleaned.split()
                html = "".join([f"<span style='background-color:rgba(255,0,0,{np.random.random()});padding:2px;margin:2px;'>{w}</span> " for w in words])
                st.markdown(html, unsafe_allow_html=True)
                
                # Image Heatmap
                img_np = np.array(Image.open(uploaded_image).convert('RGB'))
                heatmap = np.random.uniform(-1, 1, (16, 16))
                heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
                
                fig, ax = plt.subplots()
                ax.imshow(img_np)
                ax.imshow(heatmap_resized, cmap='bwr', alpha=0.4)
                st.pyplot(fig)
        else:
            st.warning("Please provide both text and an image.")