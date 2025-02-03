# Standalone script for performing image super-resolution locally
import cv2
from cv2 import dnn_superres

IMAGE_PATH = "Remember Reach.jpg"
MODEL_PATH = "FSRCNN_x2.pb"

# Initialize the OpenCV Super Resolution module
sr = dnn_superres.DnnSuperResImpl_create()

# Load the image that will be enhanced
image = cv2.imread(IMAGE_PATH)

# Import the pre-trained FSRCNN model
sr.readModel(MODEL_PATH)
sr.setModel("fsrcnn", 2)

print("Model loaded successfully")
