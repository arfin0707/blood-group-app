import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from PIL import Image
import gdown
import os

# ✅ 1. Updated MODEL_URL (Trained model contains both weights + class_to_idx)
MODEL_URL = "https://drive.google.com/uc?id=137jjKhFD9iWXBppsfUaCtHOVkNHrb1mn"

# https://drive.google.com/file/d/137jjKhFD9iWXBppsfUaCtHOVkNHrb1mn/view?usp=sharing

@st.cache_resource
def load_model():
    # ✅ 2. Download the full checkpoint with class_to_idx
    if not os.path.exists("model.pth"):
        gdown.download(MODEL_URL, "model.pth", quiet=False)

    # ✅ 3. Load architecture with pretrained weights
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 8)  # ✅ Matches your training: 8 blood classes
    )

    # ✅ 4. Load state_dict and class mapping
    checkpoint = torch.load("model.pth", map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # ✅ 5. Get class mapping (idx -> label)
    class_to_idx = checkpoint.get('class_to_idx', {
        'A+': 0, 'A-': 1, 'AB+': 2, 'AB-': 3, 'B+': 4, 'B-': 5, 'O+': 6, 'O-': 7
    })
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    return model, idx_to_class

# Load model and mapping
model, idx_to_class = load_model()

# ✅ 6. Streamlit UI
st.title("🩸 Blood Group Prediction")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    transform = ConvNeXt_Tiny_Weights.DEFAULT.transforms()
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

        # ✅ 7. Use dynamic class name mapping
        predicted_label = idx_to_class[predicted.item()]
        st.success(f"Predicted Blood Group: **{predicted_label}**")
