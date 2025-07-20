# Computer Vision in Python

This repository is a comprehensive, hands-on learning collection for mastering computer vision using Python. It combines well-organized Jupyter Notebooks for guided exploration with OpenCV and deep learning, alongside a practical project: a real-time finger counting application using your webcam.

## 🙌 Credits

This repository is based on the excellent [Pierian Data](https://www.pieriantraining.com/) computer vision curriculum, enhanced with personal projects and extensions.

## 📁 Repository Structure

```
computer-vision-in-python/
├── cv_notebook/           # Learning notebooks organized by topic
├── finger_counter_app/    # Practical application: Finger Counter
├── README.md              # You're here!
```

## 📘 Learning Notebooks

The `cv_notebook` folder is structured like a course and divided into the following sections:

### 1. **NumPy & Image Basics**

* Intro to arrays, image representation, and pixel manipulation

### 2. **Image Basics with OpenCV**

* Reading, displaying, and drawing on images
* Mouse interactions
* Assessments and solutions

### 3. **Image Processing**

* Thresholding, blurring, morphological operations, gradients, and histograms

### 4. **Video Basics**

* Live camera feed access
* Drawing and interacting with frames
* Video assessments

### 5. **Object Detection**

* Template matching, contour and edge detection
* Watershed segmentation
* Haar cascades for face detection

### 6. **Object Tracking**

* Optical flow, MeanShift, CamShift, and tracking APIs

### 7. **Deep Learning for Computer Vision**

* CNNs with Keras on MNIST and CIFAR-10
* Transfer learning, YOLOv3, and custom image classification

### 8. **Capstone Project**

* Finger counting using contour analysis and convex hulls

## 🖐️ Finger Counter App

This is a webcam-based real-time **Finger Counter** using OpenCV.

### 🔍 Features:

* ROI selection
* Background modeling with accumulated average
* Hand segmentation using image differencing
* Finger counting with contour analysis and geometric heuristics
* Live display of ROI and prediction

### ▶️ Run It

> Ensure your webcam is connected and accessible.

```bash
python finger_counter_app/finger_counter.py
```

Press **Esc** to exit the window.

## 🛠️ Requirements

Install all dependencies using one of the provided environment files:

```bash
conda env create -f cvcourse_macos.yml  # or cvcourse_linux.yml / cvcourse_windows.yml
conda activate cvcourse
```

## 📂 Dataset & Assets

All necessary media and model files are included under the `DATA/` and `06-YOLOv3/` directories:

* Images and videos for exercises
* Haar cascades and pretrained models
* Custom-trained Keras `.h5` models for classification tasks

## 👁️ YOLOv3 Object Detection

A simplified YOLOv3 implementation is included under `06-YOLOv3/`, adapted from the [`yad2k`](https://github.com/allanzelener/YAD2K) project.

* Uses `darknet53` architecture
* Includes config and weights (weights need to be downloaded separately via `.url` file)
