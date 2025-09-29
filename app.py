import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from PIL import Image
import gdown
import os

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Blood Group Prediction",
    page_icon="🩸",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown(
    """
    <style>
        /* Global background & text */
        .stApp {
            background-color: #f4f6f9 !important;
            color: #000000 !important;
        }

        /* Top header white */
        header, .st-emotion-cache-18ni7ap, .st-emotion-cache-12fmjuu {
            background-color: #ffffff !important;
            color: #000000 !important;
        }

        /* Title */
        h1 {
            color: #003366 !important;
            text-align: center !important;
            font-weight: bold !important;
        }

        /* Info card (model details) */
        .info-box {
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 15px;
            text-align: center;
            font-size: 14px;
            color: #000000;
        }

               /* =======================
           FILE UPLOADER
        ======================= */
        .stFileUploader {
            background-color: #ffffff !important;
            border: 2px solid #cccccc !important;
            border-radius: 8px !important;
            box-shadow: none !important;
            padding: 12px !important;
            color: #000000 !important;   
        }

        /* Drag-and-drop area */
        .stFileUploader div div {
            background-color: #f9f9f9 !important;
            color: #000000 !important;
            border: 1px dashed #cccccc !important;
            border-radius: 6px !important;
        }

        /* Browse button */
        .stFileUploader button {
            background-color: #e0e0e0 !important;  /* light grey */
            color: #000000 !important;             /* black text */
            border: 1px solid #999999 !important;
            border-radius: 5px !important;
            padding: 6px 20px !important;
            font-weight: 500 !important;
        }
        .stFileUploader button:hover {
            background-color: #d5d5d5 !important;
            color: #000000 !important;
        }

        /* Fix file name visibility */
        .stFileUploader label, .stFileUploader div, .stFileUploader span {
            color: #000000 !important;
        }

        # /* Browse button */
        # .stFileUploader button {
        #     background-color: #e0e0e0 !important;
        #     color: #000000 !important;
        #     border: 1px solid #999999 !important;
        #     border-radius: 5px !important;
        #     padding: 6px 20px !important;
        #     font-weight: 500 !important;
        # }
        # .stFileUploader button:hover {
        #     background-color: #d5d5d5 !important;
        #     color: #000000 !important;
        # }

        /* Success box */
        .stSuccess {
            background-color: #e6f4ea !important;
            border-left: 5px solid #2e7d32 !important;
            padding: 10px !important;
            font-size: 18px !important;
            font-weight: 500 !important;
            color: #000000 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Model Setup
# ----------------------------
MODEL_URL = "https://drive.google.com/uc?id=15HH0A3-W8aWO2epgnH8OEsDNAQsX9au8"
MODEL_FILENAME = "convnext_model_base.pth"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILENAME):
        gdown.download(MODEL_URL, MODEL_FILENAME, quiet=False)

    model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 8)
    )

    checkpoint = torch.load(MODEL_FILENAME, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    class_to_idx = checkpoint.get('class_to_idx', {
        'A+': 0, 'A-': 1, 'AB+': 2, 'AB-': 3,
        'B+': 4, 'B-': 5, 'O+': 6, 'O-': 7
    })
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    return model, idx_to_class

model, idx_to_class = load_model()

# ----------------------------
# UI
# ----------------------------
st.title("🩸 Blood Group Prediction")

# Info box
st.markdown(
    """
    <div class="info-box">
        <b>Using ConvNeXt-Base</b><br>
        Total images: 8000 <br>
        Training images: 6400 <br>
        Testing images: 1600
    </div>
    """,
    unsafe_allow_html=True
)

# File upload
uploaded_file = st.file_uploader("Upload a fingerprint image", type=["jpg", "jpeg", "png", "bmp", "svg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=250)

    transform = ConvNeXt_Base_Weights.DEFAULT.transforms()
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_label = idx_to_class[predicted.item()]
        st.success(f"Predicted Blood Group: **{predicted_label}**")
