# Image Resolution Enhancer

A Streamlit app for 2× image upscaling with OpenCV's FSRCNN super-resolution model. It also provides brightness, contrast, sharpness, saturation, and hue controls before downloading the result as a PNG.

## Run locally

```bash
python -m venv .venv
python -m pip install -r requirements.txt
streamlit run final_app.py
```

Upload a BMP, JPG, or JPEG image in the app. The included `FSRCNN_x2.pb` model must remain beside `final_app.py`.

`SuperResolution.py` is a separate script that demonstrates the upscaling pipeline with a local image. Set its `IMAGE_PATH` to an image on your computer before running it.

## Files

- `final_app.py` — Streamlit interface and enhancement controls
- `SuperResolution.py` — standalone demonstration script
- `FSRCNN_x2.pb` — pretrained 2× FSRCNN model
- `requirements.txt` — Python dependencies
