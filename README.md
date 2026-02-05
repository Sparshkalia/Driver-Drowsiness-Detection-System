# 😴 Driver Drowsiness Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![TensorFlow Lite](https://img.shields.io/badge/TensorFlow%20Lite-Edge%20AI-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

A robust, real-time drowsiness detection system designed to prevent accidents by monitoring driver alertness. This project combines **Computer Vision (OpenCV)** for efficient face/eye tracking with **Deep Learning (TensorFlow Lite)** for high-accuracy state classification.

## ✨ Key Features

*   **Real-Time Detection**: Processes video feed instantly to detect open/closed eyes.
*   **Hybrid Architecture**:
    *   **Haar Cascades**: Fast and tailored face & eye detection.
    *   **TinyML Classification**: Lightweight TFLite model (trained via Edge Impulse) for "Awake" vs. "Drowsy" prediction.
*   **Robust Filtering Logic**:
    *   **Smart ROI**: Restricts eye search to the top 60% of the face to eliminate false positives (e.g., mouth detected as eye).
    *   **Signal Smoothing**: Uses a temporal buffer (Deque) to average predictions over frames, preventing label flickering.
    *   **State Retention**: Intelligently handles momentary tracking loss (e.g., when blinking or turning head) by referencing the last known valid state.

## 🧠 Model Training & Performance

The core classification model was trained using **Edge Impulse** on a custom dataset sourced from **Kaggle**. 

*   **Dataset**: diverse set of Open vs. Closed eye images to ensure generalization.
*   **Training Platform**: Edge Impulse (TinyML).
*   **Model Architecture**: Quantized (int8) Transfer Learning model optimized for edge interaction.

### Performance Metrics

**Validation Accuracy**: 89.0%  
**Testing Accuracy**: ~92.06%


*Figure 1: Training Validation <img width="868" height="702" alt="Screenshot 2026-02-05 234814" src="https://github.com/user-attachments/assets/2958b6f0-d9da-42ab-a2bb-23596acb62e4" />
& Confusion Matrix*

*Figure 2: Model Testing <img width="1695" height="890" alt="Screenshot 2026-02-05 234903" src="https://github.com/user-attachments/assets/35be55b4-20ec-4c34-bac6-8dd43ca408fb" />
Strategy & Feature Explorer*

## 🛠️ Tech Stack

*   **Language**: Python
*   **Computer Vision**: OpenCV (`cv2`)
*   **ML Engine**: TensorFlow Lite
*   **Model Source**: Edge Impulse (Quantized int8 model for edge optimization)

## 🚀 Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/drowsiness-detection.git
    cd drowsiness-detection
    ```

2.  **Install Dependencies**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Add Assets**
    Place `training_performance.png` and `testing_results.png` in the project root to see the performance graphs in the README.

## 💻 Usage

Run the main detection script:

```bash
python drowsiness.py
```

*   **Press 'q'** to quit the application.
*   Ensure your webcam is connected and accessible.

## ⚙️ How It Works

1.  **Frame Capture**: Captures video from the webcam.
2.  **Face Detection**: Uses Haar Cascade (`haarcascade_frontalface_default`) to locate the driver's face.
3.  **ROI Extraction**: Focuses on the upper face region (eyes area) to reduce noise.
4.  **Eye Detection**: Locates eyes within the ROI using `haarcascade_eye`.
    *   *Fallback*: If face is found but eyes are missing, logic assumes eyes are likely closed (Drowsy).
5.  **Inference**:
    *   Detected eyes are cropped, resized to 96x96, and normalized.
    *   Passed to the **TFLite interpreter**.
6.  **Decision Making**:
    *   Model outputs probability of "Awake".
    *   **Smoothing**: Averages last 8 frames.
    *   **Hysteresis**: Prevents rapid toggling between states.
7.  **Alert**: Displays "AWAKE" (Green) or "DROWSY" (Red) on screen.

## 📂 Project Structure

*   `drowsiness.py`: Main application script.
*   `tflite_learn_*.tflite`: The pre-trained TinyML model.
*   `requirements.txt`: Python dependencies.
*   `edge-impulse-sdk/`: C++ SDK (if deploying to microcontrollers).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open-source.
