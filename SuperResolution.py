# Standalone script for performing image super-resolution locally
import cv2

IMAGE_PATH = "Remember Reach.jpg"

# Load the image that will be enhanced
image = cv2.imread(IMAGE_PATH)

if image is not None:
    h, w, c = image.shape
    print(f"Loaded image '{IMAGE_PATH}' with resolution {w}x{h}")
else:
    print(f"Could not load image '{IMAGE_PATH}'")
