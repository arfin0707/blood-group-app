import streamlit as st
import torch
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from PIL import Image
import gdown
import os

MODEL_URL = "https://drive.google.com/uc?id=1qtNsbahGSzvrEae9-XkSdUD48iDwOS0E"

@st.cache_resource
def load_model():
    if not os.path.exists("model.pth"):
        gdown.download(MODEL_URL, "model.pth", quiet=False)

    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(num_ftrs, 8)
    )
    state_dict = torch.load("model.pth", map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

st.title("🩸 Blood Group Prediction")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    #st.image(image, caption="Uploaded Image", use_column_width=True)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    transform = ConvNeXt_Tiny_Weights.DEFAULT.transforms()
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        #class_names = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
        class_names = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
        st.success(f"Predicted Blood Group: **{class_names[predicted.item()]}**")
