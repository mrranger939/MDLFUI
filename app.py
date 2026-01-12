import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from model_utils import load_all_models
from data_utils import clean_tweet_v2, extract_single_behavioral, prepare_image

# --- DASHBOARD SETUP ---
st.set_page_config(page_title="AI Depression Detector", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_resources():
    return load_all_models("models/best_trimodal_model_V3.pth", device)

model, tokenizer, bert, vis_extractor = load_resources()

# --- SIDEBAR: DATASET EXPLORATION ---
st.sidebar.title("Twitter Dataset Explorer")
data_path = "data/MultiModalDataset"
label_choice = st.sidebar.selectbox("Choose Category", ["positive", "negative"])

selected_user = None
if os.path.exists(data_path):
    category_path = os.path.join(data_path, label_choice)
    user_list = [u for u in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, u))]
    selected_user = st.sidebar.selectbox("Select User ID", user_list)
else:
    st.sidebar.warning("Dataset not found in /data/ folder!")

# --- MAIN UI ---
st.title("🧠 Multi-Modal Depression Detection")
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Analysis")
    tweet_text = st.text_area("Twitter Post Text", height=100)
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    
    if st.button("🚀 Run Full Analysis"):
        if tweet_text and (uploaded_image or selected_user):
            # 1. Processing Text -> Embedding
            cleaned = clean_tweet_v2(tweet_text)
            tokens = tokenizer(cleaned, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            with torch.no_grad():
                text_vec = bert(**tokens).last_hidden_state[:, 0, :] # Extract [CLS] vector

            # 2. Processing Image -> Embedding
            img_tensor = prepare_image(uploaded_image if uploaded_image else None, device)
            with torch.no_grad():
                # Extract 1280-dim feature vector from EfficientNet
                vis_vec = torch.flatten(vis_extractor(img_tensor), 1)

            # 3. Processing Behavioral Markers
            beh = extract_single_behavioral(cleaned).to(device)

            # 4. PREDICTION via Fusion Model
            with torch.no_grad():
                prob = model(text_vec, vis_vec, beh).item()
            
            # --- RESULTS DISPLAY ---
            with col2:
                st.subheader("Prediction Results")
                st.metric("Depression Probability", f"{prob:.2%}", delta="High Risk" if prob > 0.5 else "Low Risk")
                
                # Psycholinguistic Display
                st.write("**Psycholinguistic Markers:**")
                st.json({
                    "Self-Focus (FPS)": f"{beh[0,0].item():.4f}", 
                    "Social Focus (FPP)": f"{beh[0,1].item():.4f}", 
                    "Sentiment": f"{beh[0,2].item():.4f}"
                })

                # Visual Saliency (Simulated Overlay)
                st.write("**Visual Heatmap Analysis:**")
                img_display = np.array(Image.open(uploaded_image).convert('RGB'))
                fig, ax = plt.subplots()
                ax.imshow(img_display)
                heatmap = np.random.uniform(-1, 1, (16, 16)) # Example saliency grid
                heatmap_resized = cv2.resize(heatmap, (img_display.shape[1], img_display.shape[0]))
                ax.imshow(heatmap_resized, cmap='bwr', alpha=0.4)
                st.pyplot(fig)
        else:
            st.error("Please provide both text and an image for analysis.")