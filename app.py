import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance
from cv2 import dnn_superres

MODEL_PATH = "FSRCNN_x2.pb"


def load_super_resolution(scale=2):
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(MODEL_PATH)
    sr.setModel("fsrcnn", scale)
    return sr


def upscale(image, sr):
    return sr.upsample(image)


def apply_enhancements(image, brightness, contrast, sharpness, saturation, hue):
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Sharpness(image).enhance(sharpness)
    image = ImageEnhance.Color(image).enhance(saturation)
    if hue:
        hsv = np.array(image.convert("HSV"))
        hsv[..., 0] = (hsv[..., 0].astype(int) + int(hue * 255)) % 255
        image = Image.fromarray(hsv, mode="HSV").convert("RGB")
    return image


# Inject custom CSS
st.markdown(
    """
    <style>
    .main{
        background:linear-gradient(135deg,#1b1b1b,#404040);
        color:white;
    }

    h1{
        text-align:center;
        color:#ff6666;
        font-weight:700;
    }

    .stButton>button{
        background:#4CAF50;
        color:white;
        border-radius:8px;
        font-size:16px;
    }

    .stImage{
        border-radius:12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Image Controls")
brightness = st.sidebar.slider("Brightness", 0.5, 3.0, 1.0)
contrast = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0)
sharpness = st.sidebar.slider("Sharpness", 0.5, 3.0, 1.0)
saturation = st.sidebar.slider("Saturation", 0.5, 3.0, 1.0)
hue = st.sidebar.slider("Hue", -0.5, 0.5, 0.0)

st.title("Image Upscaler using Super Resolution")

uploaded = st.file_uploader(
    "Upload an image",
    type=["bmp", "jpg", "jpeg"]
)

if uploaded:
    pil_image = Image.open(uploaded)
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    st.image(pil_image, caption=f"Input Image ({uploaded.name})", use_column_width=True)
    st.write(f"**Resolution:** {pil_image.width} × {pil_image.height}")

    sr = load_super_resolution()
    upscaled_cv = upscale(cv_image, sr)
    result = Image.fromarray(cv2.cvtColor(upscaled_cv, cv2.COLOR_BGR2RGB))

    result = apply_enhancements(
        result, brightness, contrast, sharpness, saturation, hue
    )

    st.subheader("Output")
    st.image(result, caption="Enhanced Image", use_column_width=True)
    st.write(f"**Resolution:** {result.width} × {result.height}")

    # Save temporarily to offer download
    result.save("output.png")
    with open("output.png", "rb") as file:
        st.download_button(
            label="Download Image",
            data=file,
            file_name="enhanced_output.png",
            mime="image/png"
        )
else:
    st.info("Upload an image to begin.")
