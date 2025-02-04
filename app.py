import streamlit as st

st.title("Image Upscaler using Super Resolution")

uploaded = st.file_uploader(
    "Upload an image",
    type=["bmp", "jpg", "jpeg"]
)

if uploaded:
    st.info("Image uploaded successfully!")
else:
    st.info("Upload an image to begin.")
