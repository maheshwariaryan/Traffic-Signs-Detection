# Real-Time Traffic Sign Detection Project

This project aims to detect traffic signs in real-time using computer vision and a deep learning model. It employs OpenCV for real-time video processing and TensorFlow for training a convolutional neural network based on the MobileNetV2 architecture.

## Features
- **Real-time Detection**: Captures video from a webcam and detects traffic signs as they appear.
- **Deep Learning Model**: Uses a pre-trained MobileNetV2 with custom layers for accurate traffic sign classification.
- **Color Filtering**: Utilizes color-based filtering with HSV color space to isolate potential traffic sign areas.
- **Contour Analysis**: Detects contours to identify and highlight the location of traffic signs.
- **High Confidence Filtering**: Displays detected signs only if the model's prediction confidence exceeds a set threshold.

## Technologies Used
- **Python**: Programming language for implementation.
- **OpenCV**: For video capture, image processing, and contour detection.
- **NumPy**: For handling arrays and numerical operations.
- **TensorFlow & Keras**: For building and training the deep learning model.
- **MobileNetV2**: Pre-trained convolutional neural network used for transfer learning.

## Installation
- Download the main.py file. Add your dataset path in the program and train the model.
