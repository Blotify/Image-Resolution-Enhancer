# OpenCV-Streamlit based Image Upscaler and Modifier
This is a Streamlit-based web app that allows users to upscale images using Super Resolution with various image adjustment features such as brightness, contrast, sharpness, saturation, and hue control.

## Installation and Running of the Tool:
1. Download the SuperResolution.py file if you desire the native python code for the upscaler, else download app.py for the webpage-based
interface.
2. Download the FSRCNNx2.bin file where your .py file is located. If not feasible, change the value of "path" inside the .py file to match 
the download directory of the FSRCNN.bin file.
3. You will have to install all the required Python dependancies. As of the latest commit, this tool uses:

   3.a) Streamlit - "pip install streamlit"
   
   3.b) OpenCV - "pip install opencv-contrib-python"	(The contrib files are needed since dnn_superres module only exists within these contribution files)
   
   3.c) Numpy - "pip install numpy"
   
   3.d) Pillow - "pip install pillow"

5. Open the directory in Command Prompt and run this command: "streamlit run app.py" (without quotation marks)
This will open up the webpage for the tool. 

Do as you please after :)
