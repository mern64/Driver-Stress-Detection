Certainly! Here is the complete, single-page `README.md` content. You can copy the entire block below and paste it into a file named `README.md` in your project's root directory.

# Driver Stress Detection & Emotion Recognition System 🚗🛡️

This project is a Deep Learning-based application designed for **Driver Management Systems (DMS)**. It utilizes real-time facial expression recognition to monitor a driver's emotional state, specifically identifying high stress levels that could lead to road rage or unsafe driving conditions.

---

## 📌 Project Overview

Safety on the road isn't just about mechanical reliability; it's about the driver's mental state. This system provides a proactive approach to road safety by:
*   **Detecting Stress:** Identifying facial cues associated with anger, frustration, and distress.
*   **Real-time Monitoring:** Processing video feed instantly to provide immediate feedback.
*   **Preventing Road Rage:** Enabling early intervention before emotional stress escalates into dangerous driving behavior.

## 📂 Project Structure

```plain text
EMOTION RECOGNITION SYSTEM/
├── STINK3014_CaseStudy_DL_facialexpression/
│   ├── Model-Development/        # Training & Data Processing
│   │   ├── STINK3014_CaseStudy_FacialExpressionDetection_Dev.py
│   │   └── fer2013.csv           # Training Dataset
│   └── Model-Deployment/         # Real-time Application
│       ├── STINK3014_CaseStudy_FacialExpressionDetection_App.py
│       ├── stress_detector_cnn.h5 # Trained CNN Model
│       └── haarcascade_frontalface_default.xml
└── .venv/                        # Python Virtual Environment
```


## 🛠 Tech Stack

*   **Language:** Python 3.9
*   **Deep Learning:** TensorFlow / Keras (CNN Architecture)
*   **Computer Vision:** OpenCV (Haar Cascades for face detection)
*   **Data Analysis:** Pandas, NumPy, Scikit-learn
*   **Visualization:** Matplotlib

## 🚀 Getting Started

### 1. Installation
Clone the repository and set up your environment:
```shell script
git clone https://github.com/your-username/driver-stress-detection.git
cd driver-stress-detection

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install required packages
pip install tensorflow keras opencv-python pandas numpy matplotlib scikit-learn
```


### 2. Training (Optional)
To retrain the model or view the development process:
```shell script
python STINK3014_CaseStudy_DL_facialexpression/Model-Development/STINK3014_CaseStudy_FacialExpressionDetection_Dev.py
```


### 3. Running the System
To launch the real-time driver monitoring application:
```shell script
python STINK3014_CaseStudy_DL_facialexpression/Model-Deployment/STINK3014_CaseStudy_FacialExpressionDetection_App.py
```


## 🧠 How it Works

1.  **Face Detection:** The system uses a Haar Cascade classifier to locate the driver's face in the video stream.
2.  **Preprocessing:** The detected face is converted to grayscale and resized to 48x48 pixels to match the model's input requirements.
3.  **Emotion Prediction:** A Convolutional Neural Network (CNN) analyzes the facial features to predict the emotion (e.g., Angry, Sad, Neutral, Happy).
4.  **Stress Alert:** The application flags specific emotions (like Anger or high Distress) as "High Stress," indicating a risk of road rage.

## 📝 License
This project was developed as part of the **STINK3014 Case Study** for Facial Expression Detection in Driver Management Systems.

---
*With help from our Neural Network lecturer 
Associate Prof. Dr. Azizi Bin Ab Aziz .*

![](/private/var/folders/qy/yp4yyn6n1gl78c92dcsc89h80000gn/T/TemporaryItems/com.apple.Photos.NSItemProvider/uuid=2EBED645-6D94-4BFF-9D1C-9517F25C35DA&code=001&library=1&type=3&mode=1&loc=true&cap=true.mov/Screen Recording 2025-12-23 at 10.30.47 PM.mov)
