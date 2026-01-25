import streamlit as st
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from skimage.segmentation import slic
from model_utils import load_models, SHAPWrapper
from data_utils import clean_tweet_v2, get_behavioral_vector, prepare_image
from eval_utils import evaluate_whole_dataset

# --- DASHBOARD CONFIG ---
st.set_page_config(page_title="Tri-Modal Depression Diagnostic", layout="wide")
device = torch.device("cpu") # SHAP often runs more reliably on CPU for small batches

@st.cache_resource
def init():
    model, tokenizer, bert, vis_extractor = load_models("models/best_trimodal_model_V3_5.pth", device)
    # Wrap model for SHAP
    shap_model = SHAPWrapper(model)
    shap_model.eval()
    return model, shap_model, tokenizer, bert, vis_extractor

model, shap_model, tokenizer, bert, vis_extractor = init()

BEH_NAMES = [
    "Late Night Ratio", "Post Frequency", "Routine Var",
    "Self-Focus Ratio", "Collective Focus", "Sentiment Volatility",
    "Media-to-Text Ratio", "Avg Post Length", "Reply Ratio"
]

# --- UI HEADER ---
st.title("🧠 Tri-Modal Depression Interpretability Dashboard")
st.markdown("---")

# --- SIDEBAR ---
st.sidebar.header("User Context")
category = st.sidebar.selectbox("Category", ["negative", "positive"])
dataset_path = os.path.join("data", "MultiModalDataset", category)

if os.path.exists(dataset_path):
    users = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    selected_user_id = st.sidebar.selectbox("Select User ID", users)
    user_folder = os.path.join(dataset_path, selected_user_id)
    
    # LIVE METADATA VIEW
    st.sidebar.markdown("### 📊 Live Behavioral Vector")
    b_vec_preview = get_behavioral_vector(user_folder).flatten().numpy()
    for name, val in zip(BEH_NAMES, b_vec_preview):
        st.sidebar.write(f"**{name}:** `{val:.4f}`")
else:
    st.sidebar.error("Dataset not found!")

# --- MAIN UI ---
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📝 Diagnostic Input")
    tweet_text = st.text_area("Tweet Text", height=200)
    uploaded_image = st.file_uploader("Upload Image (Optional)", type=["jpg", "png", "jpeg"])
    
    analyze_btn = st.button("🚀 Run Full Diagnostic", use_container_width=True)

if analyze_btn:
    if not tweet_text:
        st.error("Please enter text content.")
    else:
        # 1. Processing
        cleaned = clean_tweet_v2(tweet_text)
        t_in = tokenizer(cleaned, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            t_vec = bert(**t_in).last_hidden_state[:, 0, :]
        
        v_vec = torch.flatten(vis_extractor(prepare_image(uploaded_image, device)), 1) if uploaded_image else torch.zeros((1, 1280)).to(device)
        b_vec = get_behavioral_vector(user_folder).to(device)

        # 2. Predict
        with torch.no_grad():
            prob = model(t_vec, v_vec, b_vec).item()
        
        # 3. REAL SHAP CALCULATION
        with st.spinner("Calculating Real-Time SHAP Values (this uses the actual model weights)..."):
            # Combine inputs exactly like the notebook
            combined_input = torch.cat((t_vec, v_vec, b_vec), dim=1).requires_grad_(True)
            
            # Create a "dummy" background (zeros) if real training data isn't loaded
            # This is standard for GradientExplainer when full dataset is missing
            background = torch.zeros((1, 2057)).to(device) 
            explainer = shap.GradientExplainer(shap_model, background)
            
            # Calculate SHAP
            shap_values = explainer.shap_values(combined_input)
            
            # Handle output format (list vs array)
            if isinstance(shap_values, list):
                instance_shap = shap_values[0].flatten()
            else:
                instance_shap = shap_values.flatten()

        # 4. DASHBOARD
        with col2:
            st.subheader("🔍 Interpretability Diagnostic Report")
            label = "Positive (Depressed)" if prob > 0.5 else "Negative (Healthy)"
            st.metric("Depression Confidence", f"{prob:.2%}", label)

            # --- PLOT 1: Modality Contribution (REAL VALUES) ---
            st.write("### 🌍 Global Modality Contribution")
            # Using exact indices from notebook
            t_sum = np.sum(np.abs(instance_shap[:768]))
            v_sum = np.sum(np.abs(instance_shap[768:2048]))
            b_sum = np.sum(np.abs(instance_shap[2048:]))
            
            fig1, ax1 = plt.subplots(figsize=(12, 4))
            ax1.bar(['Textual', 'Visual', 'Behavioral'], [t_sum, v_sum, b_sum], 
                    color=['#4e79a7', '#f28e2b', '#e15759'])
            st.pyplot(fig1)

            # --- PLOT 2: Behavioral Feature Breakdown ---
            st.write("### 📈 Behavioral Feature Impact")
            beh_vals = instance_shap[2048:2057]
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.barh(BEH_NAMES, beh_vals, color=['#d62728' if x > 0 else '#1f77b4' for x in beh_vals])
            ax2.axvline(0, color='black', lw=0.8)
            st.pyplot(fig2)

            # --- PLOT 3: Visual Saliency ---
            if uploaded_image:
                st.write("### 🖼️ Visual Saliency Map")
                img_np = np.array(Image.open(uploaded_image).convert('RGB'))
                segments = slic(img_np, n_segments=50, compactness=10)
                saliency = np.zeros(img_np.shape[:2])
                v_weights = instance_shap[768:768+50] # Map first 50 superpixels
                for i in range(50): 
                    if i < len(v_weights): saliency[segments == i] = v_weights[i]
                
                fig3, ax3 = plt.subplots(figsize=(12, 8))
                ax3.imshow(img_np)
                ax3.imshow(saliency, cmap='bwr', alpha=0.5)
                plt.axis('off')
                st.pyplot(fig3)

            # --- PLOT 4: Textual Saliency ---
            st.write("### 📝 Textual Saliency")
            tokens = cleaned.split()
            # Map first N SHAP values to N tokens
            t_weights = instance_shap[:len(tokens)]
            max_w = np.max(np.abs(t_weights)) if len(t_weights) > 0 else 1
            
            html = "<div style='line-height: 2.8; font-size: 1.2em;'>"
            for word, w in zip(tokens, t_weights / max_w):
                alpha = max(abs(w), 0.1)
                bg = f"rgba(255, 0, 0, {alpha})" if w > 0 else f"rgba(0, 0, 255, {alpha})"
                txt = "white" if alpha > 0.5 else "black"
                html += f"<span style='background-color:{bg}; color:{txt}; padding:6px 10px; margin:4px; border-radius:5px; font-weight:600;'>{word}</span> "
            st.markdown(html + "</div>", unsafe_allow_html=True)

# --- EVALUATION SECTION ---
st.markdown("---")
st.subheader("📊 Dataset Evaluation")

col_eval_btn, col_eval_res = st.columns([1, 2])

with col_eval_btn:
    st.write("Run evaluation on the entire local dataset to verify model performance.")
    if st.button("🏁 Calculate Model Accuracy"):
        with st.spinner("Running evaluation on all users in 'data/MultiModalDataset'... This may take a moment."):
            # Call the new function
            metrics = evaluate_whole_dataset(model, tokenizer, bert, vis_extractor, device)
            
            # Display Metrics nicely
            st.success("Evaluation Complete!")
            
            # Metric Cards
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
            c2.metric("Precision", f"{metrics['Precision']:.2%}")
            c3.metric("Recall", f"{metrics['Recall']:.2%}")
            c4.metric("F1 Score", f"{metrics['F1 Score']:.2%}")
            
            # Confusion Matrix
            st.write("**Confusion Matrix:**")
            cm = metrics['Confusion Matrix']
            
            # Custom HTML Table for Confusion Matrix
            st.markdown(f"""
            <table style="width:50%; text-align:center; border:1px solid #ddd;">
              <tr>
                <th></th>
                <th>Pred Negative</th>
                <th>Pred Positive</th>
              </tr>
              <tr>
                <td><b>Actual Negative</b></td>
                <td style="background-color:#e6f3ff">{cm[0][0]}</td>
                <td style="background-color:#ffe6e6">{cm[0][1]}</td>
              </tr>
              <tr>
                <td><b>Actual Positive</b></td>
                <td style="background-color:#ffe6e6">{cm[1][0]}</td>
                <td style="background-color:#e6f3ff">{cm[1][1]}</td>
              </tr>
            </table>
            """, unsafe_allow_html=True)