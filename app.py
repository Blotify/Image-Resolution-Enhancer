import cv2
import numpy as np
import streamlit as st
from PIL import Image
from cv2 import dnn_superres

MODEL_PATH = "FSRCNN_x2.pb"

st.title("Image Upscaler using Super Resolution")

uploaded = st.file_uploader(
    "Upload an image",
    type=["bmp", "jpg", "jpeg"]
)

if uploaded:
    pil_image = Image.open(uploaded)
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    st.image(pil_image, caption="Input Image", use_column_width=True)

    # Load model and upscale
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(MODEL_PATH)
    sr.setModel("fsrcnn", 2)

    upscaled_cv = sr.upsample(cv_image)
    result = Image.fromarray(cv2.cvtColor(upscaled_cv, cv2.COLOR_BGR2RGB))

    st.image(result, caption="Enhanced Image", use_column_width=True)
else:
    st.info("Upload an image to begin.")
