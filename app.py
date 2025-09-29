
import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from PIL import Image
import gdown
import os

# Custom page config
st.set_page_config(
    page_title="Blood Group Prediction",
    page_icon="🩸",
    layout="centered",
)
st.markdown(
    """
    <style>
        /* Set background color without affecting text rendering */
        .stApp {
            background-color: #f4f6f9;   /* clean light grey */
            color: black;               /* keep text crisp */
        }

        /* Title styling */
        h1 {
            color: #8B0000;  /* dark red for a formal theme */
            text-align: center;
        }

        /* Prediction result box */
        .stSuccess {
            background-color: #e6f4ea !important;  
            border-left: 5px solid #2e7d32 !important; 
            padding: 10px !important;
            font-size: 18px !important;
            font-weight: 500 !important;
            color: black !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ===============================
# Updated Model Link and Filename
# ===============================
MODEL_URL = "https://drive.google.com/uc?id=15HH0A3-W8aWO2epgnH8OEsDNAQsX9au8"
# https://drive.google.com/file/d/15HH0A3-W8aWO2epgnH8OEsDNAQsX9au8/view?usp=sharing
MODEL_FILENAME = "convnext_model_base.pth"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILENAME):
        gdown.download(MODEL_URL, MODEL_FILENAME, quiet=False)

    # Load pretrained ConvNeXt-Base
    model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.5),   # consistent with your training script
        nn.Linear(num_ftrs, 8)
    )

    # Load checkpoint
    checkpoint = torch.load(MODEL_FILENAME, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Class mapping
    class_to_idx = checkpoint.get('class_to_idx', {
        'A+': 0, 'A-': 1, 'AB+': 2, 'AB-': 3,
        'B+': 4, 'B-': 5, 'O+': 6, 'O-': 7
    })
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    return model, idx_to_class

# Load model (cached)
model, idx_to_class = load_model()

# ===============================
# Streamlit UI
# ===============================
st.title("🩸 Blood Group Prediction")

st.markdown(
    """
    <p style='font-size: 14px; color: white;'>
        Using ConvNeXt-Base <br>
        Total images: 8000 <br>
        Training images: 6400 <br>
        Testing images: 1600
    </p>
    """,
    unsafe_allow_html=True
)

# File upload
uploaded_file = st.file_uploader("Upload a fingerprint image", type=["jpg", "jpeg", "png", "bmp", "svg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    # st.image(image, caption="Uploaded Image", use_container_width=True)
    st.image(image, caption="Uploaded Image", width=250)

    # Apply ConvNeXt-Base default transforms
    transform = ConvNeXt_Base_Weights.DEFAULT.transforms()
    input_tensor = transform(image).unsqueeze(0)

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_label = idx_to_class[predicted.item()]
        st.success(f"Predicted Blood Group: **{predicted_label}**")








#convnext-tiny
# import streamlit as st
# import torch
# import torch.nn as nn
# from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
# from PIL import Image
# import gdown
# import os

# # Use consistent filename
# MODEL_URL = "https://drive.google.com/uc?id=137jjKhFD9iWXBppsfUaCtHOVkNHrb1mn"
# MODEL_FILENAME = "convnext_model.pth"

# @st.cache_resource
# def load_model():
#     if not os.path.exists(MODEL_FILENAME):
#         gdown.download(MODEL_URL, MODEL_FILENAME, quiet=False)

#     # Load the base model architecture with pretrained weights
#     model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
#     num_ftrs = model.classifier[2].in_features
#     # Replace final layer to match your 8 blood group classes
#     model.classifier[2] = nn.Sequential(
#         nn.Dropout(0.4),
#         nn.Linear(num_ftrs, 8)
#     )

#     # Load checkpoint (expects dict with model_state_dict and class_to_idx)
#     checkpoint = torch.load(MODEL_FILENAME, map_location='cpu')
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()

#     # Get class_to_idx dictionary from checkpoint or fallback default
#     class_to_idx = checkpoint.get('class_to_idx', {
#         'A+': 0, 'A-': 1, 'AB+': 2, 'AB-': 3,
#         'B+': 4, 'B-': 5, 'O+': 6, 'O-': 7
#     })

#     # Invert dict: idx -> class label
#     idx_to_class = {v: k for k, v in class_to_idx.items()}

#     return model, idx_to_class

# # Load model once (cached)
# model, idx_to_class = load_model()

# # st.title("🩸 Blood Group Prediction (used ")
# import streamlit as st

# st.title("🩸 Blood Group Prediction")

# # Small font details below the title
# st.markdown(
#     """
#     <p style='font-size: 14px; color: white;'>
#         Used ConvNeXt-Tiny <br>
#         Total images: 4480<br>
#         Number of training images: 3584<br>
#         Number of testing images: 896
#     </p>
#     """,
#     unsafe_allow_html=True
# )


# #uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "svg", "bmp"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_container_width=True)

#     transform = ConvNeXt_Tiny_Weights.DEFAULT.transforms()
#     # transform = transforms.Compose([
#     # transforms.Resize((224, 224)),
#     # transforms.RandomHorizontalFlip(),
#     # transforms.RandomRotation(10),
#     # transforms.ColorJitter(),
#     # transforms.ToTensor(),
#     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     # ])
    
#     input_tensor = transform(image).unsqueeze(0)

#     with torch.no_grad():
#         outputs = model(input_tensor)
#         _, predicted = torch.max(outputs, 1)
#         predicted_label = idx_to_class[predicted.item()]
#         st.success(f"Predicted Blood Group: **{predicted_label}**")
