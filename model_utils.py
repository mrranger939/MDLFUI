import torch
import torch.nn as nn
import streamlit as st
import os
from transformers import AutoTokenizer, AutoModel
from torchvision import models

# --- 1. UPDATED MODEL ARCHITECTURE ---
class RobustTriModalClassifier(nn.Module):
    def __init__(self, text_dim=768, visual_dim=1280, behavior_dim=9):
        super(RobustTriModalClassifier, self).__init__()
        
        # Balanced Projections (256 dims)
        self.proj_text = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.proj_visual = nn.Sequential(
            nn.Linear(visual_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.proj_behavior = nn.Sequential(
            nn.Linear(behavior_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Residual Fusion Head
        self.fusion_fc1 = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.fusion_fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Output Layer (Returns LOGITS)
        self.output = nn.Linear(128, 1)

    def forward(self, x_text, x_visual, x_behavior):
        ft = self.proj_text(x_text)
        fv = self.proj_visual(x_visual)
        fb = self.proj_behavior(x_behavior)
        
        combined = torch.cat((ft, fv, fb), dim=1)
        
        x = self.fusion_fc1(combined)
        x = self.fusion_fc2(x)
        
        return self.output(x)

# --- 2. ENSEMBLE WRAPPER (FIXED) ---
class EnsembleModel(nn.Module):
    """
    Wraps multiple trained models and averages their predictions.
    """
    def __init__(self, models_list):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models_list)

    def forward(self, t, v, b):
        # Determine device from the first model parameters
        device = next(self.models[0].parameters()).device
        
        # Ensure inputs are on correct device
        t, v, b = t.to(device), v.to(device), b.to(device)

        probs_sum = 0
        
        # [CRITICAL FIX] REMOVED 'with torch.no_grad():' 
        # We need gradients to flow for SHAP to work.
        # For simple inference, we wrap the call in no_grad externally.
        for model in self.models:
            model.eval()
            # Get logits
            logits = model(t, v, b)
            # Apply Sigmoid manually since model outputs logits
            probs_sum += torch.sigmoid(logits)
        
        # Return Average Probability
        return probs_sum / len(self.models)

# --- 3. SHAP WRAPPER ---
class SHAPWrapper(nn.Module):
    def __init__(self, ensemble_model):
        super(SHAPWrapper, self).__init__()
        self.ensemble = ensemble_model
    
    def forward(self, x):
        # Split the giant vector back into 3 parts
        t = x[:, :768]
        v = x[:, 768:2048]
        b = x[:, 2048:]
        return self.ensemble(t, v, b)

# --- 4. LOADER ---
def load_models(models_dir, device):
    hf_token = st.secrets["HF_TOKEN"]
    
    # 1. Load Ensemble Models
    models_list = []
    num_models = 5 
    
    for i in range(num_models):
        filename = f"ensemble_model_{i}.pth"
        path = os.path.join(models_dir, filename)
        
        if os.path.exists(path):
            model = RobustTriModalClassifier()
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device).eval()
            models_list.append(model)
        else:
            st.warning(f"⚠️ Could not find {filename} in {models_dir}")

    if not models_list:
        st.error("No ensemble models found! Check your directory path.")
        return None, None, None, None

    ensemble = EnsembleModel(models_list)

    # 2. MentalBERT
    m_name = "AIMH/mental-bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(m_name, token=hf_token)
    bert = AutoModel.from_pretrained(m_name, token=hf_token).to(device).eval()

    # 3. EfficientNetV2
    base_vis = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
    vis_extractor = nn.Sequential(*list(base_vis.children())[:-1]).to(device).eval()

    return ensemble, tokenizer, bert, vis_extractor