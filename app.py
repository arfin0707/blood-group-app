import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(num_ftrs, 8)  # 8 blood groups
    )
    import os
    import gdown

    MODEL_URL = 'https://drive.google.com/file/d/1lMplGh8eLhXRNSM6I2AMgyZM4s754xJE/view?usp=sharing'  # your real ID

    def download_model():
        if not os.path.exists("model.pth"):
            gdown.download(MODEL_URL, "model.pth", quiet=False)

    download_model()
    model.load_state_dict(torch.load("model.pth", map_location='cpu'))

    model.eval()
    return model

model = load_model()
class_names = ['A-', 'A+', 'AB-', 'AB+', 'B-', 'B+', 'O-', 'O+']

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# UI
st.title("🩸 Blood Group Prediction")
st.write("Upload a blood image to predict the blood group.")

file = st.file_uploader("Upload an image", type=["jpg", "png", "bmp"])

if file:
    image = Image.open(file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    st.success(f"🧬 Prediction: {class_names[pred.item()]}")
    st.info(f"Confidence: {confidence.item()*100:.2f}%")
