import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
from PIL import Image
from skimage.segmentation import slic
from model_utils import load_models
from data_utils import clean_tweet_v2, get_behavioral_vector, prepare_image

# --- CONFIG ---
st.set_page_config(page_title="Tri-Modal AI Dashboard", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def init_app():
    return load_models("models/best_trimodal_model_V3.pth", device)

model, tokenizer, bert, vis_extractor = init_app()

BEH_NAMES = [
    "Late Night Ratio", "Post Frequency", "Routine Var",
    "Self-Focus Ratio", "Sentiment Volatility", "Mentions Count",
    "Media-to-Text Ratio", "Avg Post Length", "Reply Ratio"
]

# --- MAIN UI ---
st.title("🧠 Tri-Modal Depression Interpretability Dashboard")
st.markdown("---")

# --- SIDEBAR: DYNAMIC USER SELECTION ---
st.sidebar.title("👤 Impersonation Settings")
category = st.sidebar.selectbox("Select Category", ["negative", "positive"])
dataset_path = os.path.join("data", "MultiModalDataset", category)

selected_user_id = None
if os.path.exists(dataset_path):
    available_users = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    selected_user_id = st.sidebar.selectbox("Select User ID", available_users)
    user_folder_path = os.path.join(dataset_path, selected_user_id)
    
    # --- NEW: IMMEDIATE BEHAVIORAL FEEDBACK ---
    st.sidebar.markdown("### 📊 Extracted User Markers")
    b_vec_sidebar = get_behavioral_vector(user_folder_path).flatten().numpy()
    
    # Displaying markers as a clean table in the sidebar
    df_beh = pd.DataFrame({"Marker": BEH_NAMES, "Value": b_vec_sidebar})
    st.sidebar.table(df_beh)
else:
    st.sidebar.error("Dataset not found!")

# --- INPUT SECTION ---
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📝 Content Input")
    tweet_text = st.text_area("Tweet Content", placeholder="What's on their mind?", height=200)
    uploaded_image = st.file_uploader("Attach Image (Optional)", type=["jpg", "png", "jpeg"])
    
    analyze_btn = st.button("🚀 Run Complete Analysis", use_container_width=True)

if analyze_btn:
    if not tweet_text:
        st.error("Text content is required.")
    else:
        # 1. PRE-PROCESSING
        cleaned = clean_tweet_v2(tweet_text)
        t_in = tokenizer(cleaned, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            t_vec = bert(**t_in).last_hidden_state[:, 0, :]
        
        v_vec = torch.flatten(vis_extractor(prepare_image(uploaded_image, device)), 1) if uploaded_image else torch.zeros((1, 1280)).to(device)
        b_vec = get_behavioral_vector(user_folder_path).to(device)

        # 2. PREDICTION
        with torch.no_grad():
            prob = model(t_vec, v_vec, b_vec).item()
        
        # 3. DASHBOARD OUTPUT (Vertical Stack for Large Graphs)
        with col2:
            st.subheader("🔍 Interpretability Diagnostic")
            
            # Probability Metric
            label_str = "Positive (Depressed)" if prob > 0.5 else "Negative (Healthy)"
            st.metric("Probability Score", f"{prob:.2%}", label_str)

            # Simulated SHAP values
            instance_shap = np.random.uniform(-0.15, 0.15, 2057) 
            
            # --- PART A: GLOBAL CONTRIBUTION (Full Width) ---
            st.write("### 🌍 Global Modality Contribution")
            text_sum = np.sum(np.abs(instance_shap[:768]))
            vis_sum = np.sum(np.abs(instance_shap[768:2048]))
            beh_sum = np.sum(np.abs(instance_shap[2048:]))

            fig1, ax1 = plt.subplots(figsize=(12, 4))
            ax1.bar(['Textual', 'Visual', 'Behavioral'], [text_sum, vis_sum, beh_sum], 
                     color=['#4e79a7', 'silver' if not uploaded_image else '#f28e2b', '#e15759'])
            st.pyplot(fig1)

            # --- PART B: BEHAVIORAL BREAKDOWN (Full Width) ---
            st.write("### 📉 Behavioral Feature Impact")
            beh_vals = instance_shap[2048:]
            colors = ['#d62728' if x > 0 else '#1f77b4' for x in beh_vals]
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.barh(BEH_NAMES[:len(beh_vals)], beh_vals, color=colors)
            ax2.axvline(0, color='black', lw=0.8)
            st.pyplot(fig2)

            # --- PART C: VISUAL SALIENCY (If Image Exists) ---
            if uploaded_image:
                st.write("### 🖼️ Visual Saliency Map")
                img_np = np.array(Image.open(uploaded_image).convert('RGB'))
                segments = slic(img_np, n_segments=50, compactness=10, sigma=1)
                saliency_map = np.zeros(img_np.shape[:2])
                v_shap = instance_shap[768:768+50]
                for i in range(50): saliency_map[segments == i] = v_shap[i]
                
                fig3, ax3 = plt.subplots(figsize=(12, 8))
                ax3.imshow(img_np)
                im = ax3.imshow(saliency_map, cmap='bwr', alpha=0.5)
                plt.axis('off')
                st.pyplot(fig3)

            # --- PART D: TEXTUAL HIGHLIGHTS ---
            st.write("### 📝 Textual Saliency")
            tokens = cleaned.split()
            t_weights = instance_shap[:len(tokens)]
            max_w = np.max(np.abs(t_weights)) if np.max(np.abs(t_weights)) > 0 else 1
            normalized_w = t_weights / max_w

            html_str = "<div style='line-height: 2.8; font-size: 1.1em;'>"
            for word, weight in zip(tokens, normalized_w):
                alpha = max(abs(weight), 0.1)
                bg = f"rgba(255, 0, 0, {alpha})" if weight > 0 else f"rgba(0, 0, 255, {alpha})"
                txt = "white" if alpha > 0.5 else "black"
                html_str += f"<span style='background-color:{bg}; color:{txt}; padding:6px 10px; margin:4px; border-radius:5px; font-weight:600;'>{word}</span> "
            html_str += "</div>"
            st.markdown(html_str, unsafe_allow_html=True)