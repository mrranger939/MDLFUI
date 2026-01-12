import torch
import torch.nn as nn
import streamlit as st
from transformers import AutoTokenizer, AutoModel
from torchvision import models

class RobustTriModalClassifier(nn.Module):
    def __init__(self, text_dim=768, visual_dim=1280, behavior_dim=9):
        super(RobustTriModalClassifier, self).__init__()
        self.input_dim = text_dim + visual_dim + behavior_dim
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x_text, x_visual, x_behavior):
        combined = torch.cat((x_text, x_visual, x_behavior), dim=1)
        return self.network(combined)

def load_models(model_path, device):
    # Retrieve token from .streamlit/secrets.toml
    hf_token = st.secrets["HF_TOKEN"]
    
    # 1. Load your trained Fusion Model (cell 23)
    model = RobustTriModalClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()

    # 2. Load MentalBERT (cell 12)
    m_name = "AIMH/mental-bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(m_name, token=hf_token)
    bert = AutoModel.from_pretrained(m_name, token=hf_token).to(device).eval()

    # 3. Load EfficientNetV2 for Visual Extraction (cell 19)
    base_vis = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
    vis_extractor = nn.Sequential(*list(base_vis.children())[:-1]).to(device).eval()

    return model, tokenizer, bert, vis_extractor