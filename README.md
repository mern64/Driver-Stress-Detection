# Emotion Recognition & Stress Detection System

This project is a Deep Learning-based application designed to recognize facial expressions and classify them into two primary emotional states: **Stress** and **Calm**. By leveraging Convolutional Neural Networks (CNN) and real-time computer vision, the system provides an automated way to monitor emotional well-being.

## 🚀 The Application: How it Works

The system is built using a structured pipeline that moves from raw data to a real-time deployment:

### 1. Data Processing
The model is trained using the **FER2013** dataset. The application simplifies the 7 original emotion categories (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) into a binary classification:
*   **Stress:** Angry, Disgust, Fear, and Sad.
*   **Calm:** Happy, Surprise, and Neutral.

### 2. The Neural Network Architecture
The core of the system is a **Convolutional Neural Network (CNN)** built with TensorFlow/Keras. The architecture includes:
*   **Convolutional Layers:** To extract spatial features from facial images (edges, shapes, and textures).
*   **MaxPooling:** To reduce dimensionality and focus on the most important features.
*   **Dropout Layer:** To prevent overfitting, ensuring the model generalizes well to new faces.
*   **Softmax Output:** To provide a probability-based classification between "Stress" and "Calm."

### 3. Model Deployment
The project includes a deployment script (`STINK3014_CaseStudy_FacialExpressionDetection_App.py`) that uses **OpenCV** and a **Haar Cascade classifier** to detect faces in real-time via a webcam, then feeds those frames into the trained `.h5` model for instant prediction.

## 🛠 Tech Stack
*   **Language:** Python 3.9
*   **Deep Learning:** TensorFlow / Keras
*   **Computer Vision:** OpenCV
*   **Data Science:** NumPy, Pandas, Scikit-learn
*   **Visualization:** Matplotlib

## ✨ Benefits of this App

*   **Real-time Monitoring:** Capable of detecting emotional shifts instantly via live video feed.
*   **Stress Management:** Can be integrated into workplace wellness programs or driver monitoring systems to identify high-stress levels and suggest breaks.
*   **Objective Analysis:** Provides a data-driven approach to understanding emotions, removing human subjectivity in sentiment analysis.
*   **Automation:** Reduces the need for manual observation in research or safety-critical environments.

## 📁 Project Structure
```plain text
EMOTION RECOGNITION SYSTEM/
├── Model-Development/
│   ├── STINK3014_..._Dev.py       # Training script and model architecture
│   └── EmotionRecognition... .md   # Documentation for the development process
├── Model-Deployment/
│   ├── STINK3014_..._App.py       # Real-time application script
│   ├── stress_detector_cnn.h5     # The trained brain of the system
│   └── haarcascade_..._default.xml # OpenCV face detection configuration
└── .gitignore                     # Ensures large datasets aren't tracked in VCS
```


## ⚙️ How to Run
1.  **Install Dependencies:**
```shell script
pip install tensorflow opencv-python pandas scikit-learn matplotlib
```

2.  **Train (Optional):** Run the script in the `Model-Development` folder to retrain the model.
3.  **Deploy:** Run the script in `Model-Deployment` to start the real-time recognition app.

*** 

*Developed as part of the STINK3014 Case Study.*

## 📺 Demo
Check out the system in action! Below is a demonstration of the real-time facial expression detection and stress classification:

https://github.com/user-attachments/assets/ece3d792-0b94-4cba-a0d8-3f42526ed109


> **Note:** The demo highlights the system's ability to track facial movements and update the "Stress" vs "Calm" status dynamically as the user's expression changes.
