import streamlit as st
from PIL import Image
import torch
import numpy as np

# Hypothetical imports - replace with actual classes if they differ
from transformers import SAM2Processor, SAM2Model

# Load the model and processor
model = SAM2Model.from_pretrained("meta/sam2")
processor = SAM2Processor.from_pretrained("meta/sam2")

st.title("SAM2 Image Segmentation")

# Upload image
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Process the image
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract masks (assuming masks is a tensor and has shape [batch_size, channels, height, width])
    masks = outputs.masks
    
    # Convert the first mask to a numpy array
    mask_image = masks[0].cpu().numpy()

    # Normalize and convert to a PIL image
    mask_image = (mask_image - mask_image.min()) / (mask_image.max() - mask_image.min())  # Normalize to [0, 1]
    mask_image = (mask_image * 255).astype(np.uint8)  # Convert to [0, 255]
    mask_image = Image.fromarray(mask_image)

    # Display the segmented image
    st.image(mask_image, caption="Segmented Image", use_column_width=True)
