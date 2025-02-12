import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance
from cv2 import dnn_superres


# Location of the pre-trained FSRCNN model
MODEL_PATH = "FSRCNN_x2.pb"


# Creates and configures the super-resolution engine
def load_super_resolution(scale=2):
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(MODEL_PATH)
    sr.setModel("fsrcnn", scale)
    return sr


# Performs image upscaling using the loaded model
def upscale(image, sr):
    return sr.upsample(image)


# Applies all user-selected image enhancement operations
def apply_enhancements(image, brightness, contrast, sharpness, saturation, hue):

    # Modify image brightness
    image = ImageEnhance.Brightness(image).enhance(brightness)

    # Adjust the overall contrast
    image = ImageEnhance.Contrast(image).enhance(contrast)

    # Control edge clarity and fine details
    image = ImageEnhance.Sharpness(image).enhance(sharpness)

    # Increase or decrease color intensity
    image = ImageEnhance.Color(image).enhance(saturation)

    # Shift hue values by manipulating the HSV color space
    if hue:
        hsv = np.array(image.convert("HSV"))
        hsv[..., 0] = (hsv[..., 0].astype(int) + int(hue * 255)) % 255
        image = Image.fromarray(hsv, mode="HSV").convert("RGB")

    return image


# Configure the webpage before rendering any elements
st.set_page_config(
    page_title="Image Upscaler",
    page_icon="🖼️",
    layout="wide"
)

# Inject custom CSS to modify Streamlit's default appearance
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

# Sidebar controls used to customize the output image
st.sidebar.header("Image Controls")

brightness = st.sidebar.slider("Brightness", 0.5, 3.0, 1.0)
contrast = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0)
sharpness = st.sidebar.slider("Sharpness", 0.5, 3.0, 1.0)
saturation = st.sidebar.slider("Saturation", 0.5, 3.0, 1.0)
hue = st.sidebar.slider("Hue", -0.5, 0.5, 0.0)

# Main heading displayed on the application
st.title("Image Upscaler using Super Resolution")

# Accept image files from the user
uploaded = st.file_uploader(
    "Upload an image",
    type=["bmp", "jpg", "jpeg"]
)

# Begin processing only after an image is provided
if uploaded:

    # Read the uploaded image using Pillow
    pil_image = Image.open(uploaded)

    # Convert from PIL's RGB format to OpenCV's BGR format
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Preview the uploaded image
    st.image(
        pil_image,
        caption=f"Input Image ({uploaded.name})",
        use_column_width=True,
    )

    # Display the original dimensions
    st.write(
        f"**Resolution:** {pil_image.width} × {pil_image.height}"
    )

    # Load the super-resolution network
    sr = load_super_resolution()

    # Generate the higher-resolution image
    upscaled_cv = upscale(cv_image, sr)

    # Convert the OpenCV output back into a PIL image
    result = Image.fromarray(
        cv2.cvtColor(upscaled_cv, cv2.COLOR_BGR2RGB)
    )

    # Apply the selected enhancement settings
    result = apply_enhancements(
        result,
        brightness,
        contrast,
        sharpness,
        saturation,
        hue,
    )

    # Display the processed image
    st.subheader("Output")
    st.image(
        result,
        caption="Enhanced Image",
        use_column_width=True,
    )

    # Show the resolution after upscaling
    st.write(
        f"**Resolution:** {result.width} × {result.height}"
    )

    # Create a descriptive filename using the selected parameters
    filename = (
        f"SuperRes_"
        f"b{brightness:.2f}_"
        f"c{contrast:.2f}_"
        f"s{sharpness:.2f}_"
        f"sat{saturation:.2f}_"
        f"h{hue:.2f}.png"
    )

    # Save the processed image temporarily
    result.save(filename)

    # Allow the processed image to be downloaded
    with open(filename, "rb") as image_file:
        st.download_button(
            label="Download Image",
            data=image_file,
            file_name=filename,
            mime="image/png",
        )

# Display a message before any image is uploaded
else:
    st.info("Upload an image to begin.")