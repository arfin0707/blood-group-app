import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from PIL import Image
import gdown
import os

# Use consistent filename
MODEL_URL = "https://drive.google.com/uc?id=137jjKhFD9iWXBppsfUaCtHOVkNHrb1mn"
MODEL_FILENAME = "convnext_model.pth"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILENAME):
        gdown.download(MODEL_URL, MODEL_FILENAME, quiet=False)

    # Load the base model architecture with pretrained weights
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    num_ftrs = model.classifier[2].in_features
    # Replace final layer to match your 8 blood group classes
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 8)
    )

    # Load checkpoint (expects dict with model_state_dict and class_to_idx)
    checkpoint = torch.load(MODEL_FILENAME, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get class_to_idx dictionary from checkpoint or fallback default
    class_to_idx = checkpoint.get('class_to_idx', {
        'A+': 0, 'A-': 1, 'AB+': 2, 'AB-': 3,
        'B+': 4, 'B-': 5, 'O+': 6, 'O-': 7
    })

    # Invert dict: idx -> class label
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    return model, idx_to_class

# Load model once (cached)
model, idx_to_class = load_model()

st.title("🩸 Blood Group Prediction")

#uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "svg", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    transform = ConvNeXt_Tiny_Weights.DEFAULT.transforms()
    # transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),
    # transforms.ColorJitter(),
    # transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_label = idx_to_class[predicted.item()]
        st.success(f"Predicted Blood Group: **{predicted_label}**")
