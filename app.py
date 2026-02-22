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
from transformers import pipeline

# --- DASHBOARD CONFIG ---
st.set_page_config(page_title="Tri-Modal Depression Diagnostic", layout="wide")
device = torch.device("cpu") # SHAP runs more reliably on CPU for small batches

@st.cache_resource
def init():
    ensemble_model, tokenizer, bert, vis_extractor = load_models("models/ensemble/", device)
    shap_model = SHAPWrapper(ensemble_model)
    shap_model.eval()
    
    # --- LOAD LOCAL LLM FOR REPORTS ---
    # Qwen 1.5B is highly capable but small enough to run locally.
    # If it's too slow on your CPU, change it to "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    st.sidebar.text("Loading local LLM...")
    local_llm = pipeline(
        "text-generation", 
        model="Qwen/Qwen2.5-1.5B-Instruct", 
        device=-1 if device.type == "cpu" else 0 # -1 forces CPU, 0 uses GPU
    )
    
    return ensemble_model, shap_model, tokenizer, bert, vis_extractor, local_llm

# Unpack the new local_llm
model, shap_model, tokenizer, bert, vis_extractor, local_llm = init()

BEH_NAMES = [
    "Late Night Ratio", "Post Frequency", "Routine Var",
    "Self-Focus Ratio", "Collective Focus", "Sentiment Volatility",
    "Media-to-Text Ratio", "Avg Post Length", "Reply Ratio"
]

def extract_top_shap_features(instance_shap, tokens, beh_names):
    # Calculate Modality Percentages
    t_sum = np.sum(np.abs(instance_shap[:768]))
    v_sum = np.sum(np.abs(instance_shap[768:2048]))
    b_sum = np.sum(np.abs(instance_shap[2048:]))
    total = t_sum + v_sum + b_sum if (t_sum + v_sum + b_sum) > 0 else 1
    
    modality_pct = {
        "Text": f"{(t_sum/total):.1%}",
        "Visual": f"{(v_sum/total):.1%}",
        "Behavioral": f"{(b_sum/total):.1%}"
    }

    # Extract Top Words
    t_weights = instance_shap[:len(tokens)]
    word_impacts = list(zip(tokens, t_weights))
    word_impacts.sort(key=lambda x: x[1], reverse=True)
    top_words = [w[0] for w in word_impacts if w[1] > 0][:5]

    # Extract Top Behaviors
    beh_vals = instance_shap[2048:2057]
    beh_impacts = list(zip(beh_names, beh_vals))
    beh_impacts.sort(key=lambda x: x[1], reverse=True)
    top_behaviors = [b[0] for b in beh_impacts if b[1] > 0][:3]
    
    return modality_pct, top_words, top_behaviors

def create_local_llm_messages(prob, modality_pct, top_words, top_behaviors, has_image):
    prediction = "Positive for Depression Risk" if prob > 0.52 else "Negative (Healthy)"
    
    # We use a chat format (system/user messages) which works best for instruct models
    messages = [
        {
            "role": "system", 
            "content": "You are a clinical data summarizer. Write a single, concise, professional paragraph explaining why the AI made its prediction based ONLY on the data provided. Do not invent medical advice."
        },
        {
            "role": "user", 
            "content": f"""
Prediction: {prediction}
Confidence: {prob:.1%}

Key Factors:
- Modality Weights: Text ({modality_pct['Text']}), Images ({modality_pct['Visual']}), Behavior ({modality_pct['Behavioral']}).
- Top Text Triggers: {', '.join(top_words) if top_words else 'None'}.
- Top Behavioral Triggers: {', '.join(top_behaviors) if top_behaviors else 'None'}.
- Visuals: {'Analyzed and contributed to the decision.' if has_image else 'No image provided.'}

Write the summary paragraph now:"""
        }
    ]
    return messages

# --- UI HEADER ---
st.title("🧠 Tri-Modal Ensemble Diagnostic Dashboard")
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
        # 1. Processing Inputs
        cleaned = clean_tweet_v2(tweet_text)
        t_in = tokenizer(cleaned, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            t_vec = bert(**t_in).last_hidden_state[:, 0, :]
        
        v_vec = torch.flatten(vis_extractor(prepare_image(uploaded_image, device)), 1) if uploaded_image else torch.zeros((1, 1280)).to(device)
        b_vec = get_behavioral_vector(user_folder).to(device)

        # 2. Ensemble Prediction
        # The 'model' here is the EnsembleModel class from model_utils
        # It handles averaging internally.
        with torch.no_grad():
            prob = model(t_vec, v_vec, b_vec).item()
        
        # 3. REAL SHAP CALCULATION (Ensemble Aware)
        with st.spinner("Calculating SHAP Values for Ensemble..."):
            # Combine inputs exactly like the notebook
            combined_input = torch.cat((t_vec, v_vec, b_vec), dim=1).requires_grad_(True)
            
            # Create a "dummy" background (zeros)
            # For ensembles, GradientExplainer is robust enough with zero background
            background = torch.zeros((1, 2057)).to(device) 
            
            # Using GradientExplainer on the SHAPWrapper (which wraps the Ensemble)
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
            # Using 0.52 threshold from notebook optimization
            label = "Positive (Depressed)" if prob > 0.52 else "Negative (Healthy)"
            st.metric("Ensemble Confidence", f"{prob:.2%}", label)

            # --- PLOT 1: Global Modality Contribution ---
            st.write("### 🌍 Global Modality Contribution")
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
                try:
                    img_np = np.array(Image.open(uploaded_image).convert('RGB'))
                    # Ensure image is large enough for slic
                    if img_np.shape[0] > 50 and img_np.shape[1] > 50:
                         segments = slic(img_np, n_segments=50, compactness=10, sigma=1)
                         saliency = np.zeros(img_np.shape[:2])
                         v_weights = instance_shap[768:2048]
                         
                         # Map SHAP weights to segments (Approximation)
                         # We map the first 50 visual features to the 50 segments
                         for i in range(50): 
                             if i < len(v_weights): 
                                 saliency[segments == i] = v_weights[i]
                             
                         fig3, ax3 = plt.subplots(figsize=(12, 8))
                         ax3.imshow(img_np)
                         ax3.imshow(saliency, cmap='bwr', alpha=0.5)
                         plt.axis('off')
                         st.pyplot(fig3)
                    else:
                        st.warning("Image too small for Saliency Map generation.")
                except Exception as e:
                    st.error(f"Could not generate saliency map: {e}")

            # --- PLOT 4: Textual Saliency ---
            st.write("### 📝 Textual Saliency")
            tokens = cleaned.split()
            t_weights = instance_shap[:len(tokens)]
            max_w = np.max(np.abs(t_weights)) if len(t_weights) > 0 else 1
            
            html = "<div style='line-height: 2.8; font-size: 1.2em;'>"
            for word, w in zip(tokens, t_weights / max_w):
                alpha = max(abs(w), 0.1)
                bg = f"rgba(255, 0, 0, {alpha})" if w > 0 else f"rgba(0, 0, 255, {alpha})"
                txt = "white" if alpha > 0.5 else "black"
                html += f"<span style='background-color:{bg}; color:{txt}; padding:6px 10px; margin:4px; border-radius:5px; font-weight:600;'>{word}</span> "
            st.markdown(html + "</div>", unsafe_allow_html=True)
    
    # --- PLOT 4: Textual Saliency ---
            # ... (Your existing Text Saliency code here) ...
            
            st.markdown("---")
            st.write("### 🤖 Local AI Diagnostic Summary")
            
            # Extract plain English facts from SHAP
            modality_pct, top_words, top_behaviors = extract_top_shap_features(instance_shap, tokens, BEH_NAMES)
            
            # Format messages for the local model
            messages = create_local_llm_messages(prob, modality_pct, top_words, top_behaviors, uploaded_image is not None)
            
            with st.spinner("Local AI is writing the report (this may take 10-20 seconds on CPU)..."):
                try:
                    # Run the local LLM
                    output = local_llm(
                        messages, 
                        max_new_tokens=150, 
                        temperature=0.3, # Low temp keeps it focused and factual
                        do_sample=True
                    )
                    
                    # Extract the generated text
                    report = output[0]['generated_text'][-1]['content']
                    
                    st.success(report)
                except Exception as e:
                    st.error(f"Local text generation failed: {str(e)}")

# --- EVALUATION SECTION ---
st.markdown("---")
st.subheader("📊 Dataset Evaluation")

col_eval_btn, col_eval_res = st.columns([1, 2])

with col_eval_btn:
    st.write("Run evaluation on the entire local dataset to verify Ensemble performance.")
    if st.button("🏁 Calculate Ensemble Accuracy"):
        with st.spinner("Running voting evaluation on all users... This may take a moment."):
            # Call the updated function from eval_utils
            metrics = evaluate_whole_dataset(model, tokenizer, bert, vis_extractor, device)
            
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