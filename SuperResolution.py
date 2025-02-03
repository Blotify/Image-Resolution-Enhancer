# Standalone script for performing image super-resolution locally

import cv2
from cv2 import dnn_superres


# File locations for the input image and trained model
IMAGE_PATH = "Remember Reach.jpg"
MODEL_PATH = "FSRCNN_x2.pb"
OUTPUT_PATH = "SuperResOutput.png"


# Initialize the OpenCV Super Resolution module
sr = dnn_superres.DnnSuperResImpl_create()

# Load the image that will be enhanced
image = cv2.imread(IMAGE_PATH)

# Import the pre-trained FSRCNN model
sr.readModel(MODEL_PATH)

# Uncomment the following lines to use CUDA acceleration (if available)
# sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Specify the model architecture and corresponding scaling factor
sr.setModel("fsrcnn", 2)

# Produce the higher-resolution image
upscaled_image = sr.upsample(image)

# Store the processed image on disk
cv2.imwrite(OUTPUT_PATH, upscaled_image)

print(f"Upscaled image saved as '{OUTPUT_PATH}'")