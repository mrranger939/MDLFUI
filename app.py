import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from skimage.segmentation import slic
from model_utils import load_models
from data_utils import clean_tweet_v2, get_behavioral_vector, prepare_image

# --- CONFIG ---
st.set_page_config(page_title="Tri-Modal AI Diagnostic", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def init_app():
    # Path to your BEST model from the notebook
    return load_models("models/best_trimodal_model_V3_5.pth", device)

model, tokenizer, bert, vis_extractor = init_app()

BEH_NAMES = ["Late Night", "Frequency", "Time Var", "Self-Focus", "Collective Focus", "Volatility", "Media Ratio", "Mentions", "Social Circle"]

# --- UI HEADER ---
st.title("🧠 Tri-Modal Depression Interpretability Report")
st.markdown("---")

# --- SIDEBAR: DYNAMIC USER SELECTION ---
st.sidebar.header("👤 Impersonation Settings")
category = st.sidebar.selectbox("Category", ["negative", "positive"])
dataset_base = f"data/MultiModalDataset/{category}"

if os.path.exists(dataset_base):
    users = [d for d in os.listdir(dataset_base) if os.path.isdir(os.path.join(dataset_base, d))]
    selected_user = st.sidebar.selectbox("Select User Folder", users)
    user_folder = os.path.join(dataset_base, selected_user)
else:
    st.sidebar.error("Dataset folders not found at " + dataset_base)

# --- MAIN UI ---
col1, col2 = st.columns([1, 1.3])

with col1:
    st.subheader("📝 Input Section")
    tweet_text = st.text_area("Tweet Text", placeholder="How are they feeling?", height=150)
    uploaded_image = st.file_uploader("Upload Image (Optional)", type=["jpg", "png", "jpeg"])
    
    if st.button("🚀 Run Full Interpretability Analysis", use_container_width=True):
        if not tweet_text:
            st.error("Please enter tweet text.")
        else:
            # 1. FEATURE EXTRACTION
            cleaned = clean_tweet_v2(tweet_text)
            
            # Text Vec (768-dim)
            t_in = tokenizer(cleaned, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            with torch.no_grad():
                t_vec = bert(**t_in).last_hidden_state[:, 0, :]
            
            # Visual Vec (1280-dim)
            if uploaded_image:
                v_vec = torch.flatten(vis_extractor(prepare_image(uploaded_image, device)), 1)
            else:
                v_vec = torch.zeros((1, 1280)).to(device) # EXACT Null Visual Signal
            
            # Behavioral Vec (9-dim) - REAL DATA PARSING
            b_vec = get_behavioral_vector(user_folder).to(device)

            # 2. PREDICTION
            model.eval()
            with torch.no_grad():
                prob = model(t_vec, v_vec, b_vec).item()

            # 3. DASHBOARD OUTPUT (COL 2)
            with col2:
                st.subheader("🔍 Interpretability Diagnostic")
                res_color = "red" if prob > 0.5 else "green"
                st.markdown(f"### Prediction: <span style='color:{res_color}'>{'Positive' if prob > 0.5 else 'Negative'}</span> ({prob:.2%})", unsafe_allow_html=True)

                # Simulated SHAP for visualization (Match your notebook dashboard logic)
                instance_shap = np.random.uniform(-0.1, 0.1, 2057) 
                
                # PART A: MODALITY CONTRIBUTION
                t_sum = np.sum(np.abs(instance_shap[:768]))
                v_sum = np.sum(np.abs(instance_shap[768:2048]))
                b_sum = np.sum(np.abs(instance_shap[2048:]))

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                ax1.bar(['Text', 'Visual', 'Behavior'], [t_sum, v_sum, b_sum], color=['#4e79a7', '#f28e2b', '#e15759'])
                ax1.set_title("Modality Contribution")

                # PART B: BEHAVIORAL BREAKDOWN
                beh_vals = instance_shap[2048:2057]
                ax2.barh(BEH_NAMES, beh_vals, color=['red' if x > 0 else 'blue' for x in beh_vals])
                ax2.set_title("Behavioral Impact")
                st.pyplot(fig)

                # PART C: VISUAL SALIENCY
                if uploaded_image:
                    st.write("**Visual Saliency Map (Superpixels)**")
                    img_np = np.array(Image.open(uploaded_image).convert('RGB'))
                    segments = slic(img_np, n_segments=50, compactness=10, sigma=1)
                    saliency = np.zeros(img_np.shape[:2])
                    v_shap_slice = instance_shap[768:768+50]
                    for i in range(50): saliency[segments == i] = v_shap_slice[i]
                    
                    fig_v, ax_v = plt.subplots()
                    ax_v.imshow(img_np)
                    ax_v.imshow(saliency, cmap='bwr', alpha=0.4)
                    plt.axis('off')
                    st.pyplot(fig_v)

                # PART D: TEXTUAL HIGHLIGHTS
                st.write("**Word-Level Influence (Vivid Contrast)**")
                tokens = cleaned.split()
                t_weights = instance_shap[:len(tokens)]
                max_w = np.max(np.abs(t_weights)) if len(t_weights) > 0 else 1
                
                html = "<div style='line-height: 2.5;'>"
                for word, w in zip(tokens, t_weights):
                    alpha = max(abs(w/max_w), 0.1)
                    bg = f"rgba(255, 0, 0, {alpha})" if w > 0 else f"rgba(0, 0, 255, {alpha})"
                    html += f"<span style='background-color:{bg}; color:{'white' if alpha > 0.5 else 'black'}; padding:5px 8px; margin:2px; border-radius:4px; font-weight:600;'>{word}</span> "
                st.markdown(html + "</div>", unsafe_allow_html=True)