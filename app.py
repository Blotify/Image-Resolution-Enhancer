import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance
from cv2 import dnn_superres

MODEL_PATH = "FSRCNN_x2.pb"

st.sidebar.header("Image Controls")
brightness = st.sidebar.slider("Brightness", 0.5, 3.0, 1.0)
contrast = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0)
sharpness = st.sidebar.slider("Sharpness", 0.5, 3.0, 1.0)

st.title("Image Upscaler using Super Resolution")

uploaded = st.file_uploader(
    "Upload an image",
    type=["bmp", "jpg", "jpeg"]
)

if uploaded:
    pil_image = Image.open(uploaded)
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    st.image(pil_image, caption="Input Image", use_column_width=True)

    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(MODEL_PATH)
    sr.setModel("fsrcnn", 2)

    upscaled_cv = sr.upsample(cv_image)
    result = Image.fromarray(cv2.cvtColor(upscaled_cv, cv2.COLOR_BGR2RGB))

    # Apply brightness and contrast enhancements
    result = ImageEnhance.Brightness(result).enhance(brightness)
    result = ImageEnhance.Contrast(result).enhance(contrast)

    st.image(result, caption="Enhanced Image", use_column_width=True)
else:
    st.info("Upload an image to begin.")
