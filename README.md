
# 🚆 Detection-of-defective-railway-tracks

### AI-Powered Railway Track Inspection Using Deep Learning

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red?logo=streamlit)](https://railguard-ai-capestone.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-orange?logo=pytorch)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Interface-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)

**RailGuard AI** is a deep learning-based railway track inspection project designed to classify railway track images into five categories. It uses trained convolutional neural network models to support the identification of potential track defects from uploaded images.

🌐 **Live Application:**  
### 👉 [Launch Detection-of-defective-railway-tracks
](https://railguard-ai-capestone.streamlit.app/)

---

## 📌 Project Overview

Railway track inspection is essential for maintaining railway infrastructure and identifying potential defects. RailGuard AI explores how computer vision and deep learning can assist with railway track condition classification.

Users can upload a railway track image through the Streamlit application and select one of two trained models to obtain a predicted class and confidence scores.

The application also flags predictions with low confidence to encourage further inspection.

## 🎯 Objectives

- Classify railway track images into five categories.
- Apply deep learning models to railway track image analysis.
- Compare MobileNetV2 and EfficientNetB0 for image classification.
- Display prediction confidence and alternative class scores.
- Provide a simple, interactive interface for image-based inference.
- Encourage further inspection when a prediction is uncertain.

## 🧠 Defect Categories

The application supports the following five classes:

| Category | Description |
|---|---|
| Broken Track | Images classified as showing a broken or damaged track |
| Loose Bolts | Images classified as showing loose track fasteners |
| Missing Parts | Images classified as showing missing track components |
| Rust | Images classified as showing rust or corrosion |
| Normal Track | Images classified as normal track without an identified defect |

## 🤖 Deep Learning Models

RailGuard AI includes two trained image classification models:

### 1. MobileNetV2
A lightweight convolutional neural network architecture suitable for image classification applications where computational efficiency is important.

### 2. EfficientNetB0
A convolutional neural network architecture designed to balance model accuracy and computational efficiency.

Users can select either model from the application interface.

## ✨ Key Features

- **Image Upload:** Upload railway track images in JPG, JPEG, or PNG format.
- **Two Model Options:** Choose MobileNetV2 or EfficientNetB0.
- **Automated Classification:** Predict one of the five supported categories.
- **Confidence Scores:** View the model's confidence for the predicted class.
- **Top-3 Predictions:** Review the three highest-scoring class predictions.
- **Class Probability Scores:** Explore scores for all supported classes.
- **Low-Confidence Warning:** Flag uncertain predictions for additional review.
- **Interactive Web Interface:** Use the application through Streamlit.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning inference |
| Torchvision | Image transformation and model utilities |
| Pillow | Image loading and preprocessing |
| NumPy | Numerical operations |
| Streamlit | Interactive web application |
| Git and GitHub | Version control and source code hosting |
| Streamlit Community Cloud | Application deployment |

## 🏗️ Project Structure

```text
RailGuard-AI/
│
├── app.py
├── requirements.txt
│
├── railguard/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── metrics.py
│   ├── models.py
│   └── predict.py
│
├── models/
│   ├── mobilenet_v2_best.pt
│   └── efficientnet_b0_best.pt
│
├── public/
│   └── models/
│
├── scripts/
├── configs/
├── tests/
├── docs/
│
└── README.md
```

*Note: The structure above highlights the main application, model, and supporting directories. Additional files may be present in the repository.*

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.11 or 3.12
- pip
- Git

### 1. Clone the repository

```bash
git clone https://github.com/Salma1612/RailGuard-AI.git
cd RailGuard-AI
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser. If it does not open automatically, use the local URL displayed in the terminal.

## 📊 Model Inference Workflow

1. Open the RailGuard AI application.
2. Upload a railway track image.
3. Select MobileNetV2 or EfficientNetB0.
4. Click **Analyze Track**.
5. View the predicted class and confidence score.
6. Review the top-three predictions and class scores.
7. Follow the warning if the prediction has low confidence.

## 🔍 Prediction and Confidence

The application uses the selected trained checkpoint to generate class scores. It applies softmax to produce scores that sum to one across the supported classes.

The application displays:
- Predicted class
- Top-1 confidence score
- Top-three class predictions
- Scores for all classes
- Inference time
- Low-confidence warning, when applicable

**Important:** Softmax scores are not necessarily calibrated probabilities. A high score does not guarantee that a prediction is correct.

## 🚀 Deployment

RailGuard AI is deployed using Streamlit Community Cloud.

🌐 **Live Demo:** [https://railguard-ai-capestone.streamlit.app/](https://railguard-ai-capestone.streamlit.app/)

The application source code and trained model checkpoints are maintained in this GitHub repository.

## 🔮 Future Enhancements

- Expand the dataset with more diverse railway track images.
- Evaluate model performance across different lighting and weather conditions.
- Improve defect localization and visual explanations.
- Add model comparison and evaluation dashboards.
- Explore deployment options for edge devices and railway inspection systems.
- Improve robustness and validation on real-world railway images.

## ⚠️ Limitations and Safety Disclaimer

RailGuard AI is an educational and research prototype for image classification. Its predictions may be affected by image quality, lighting, camera angle, dataset limitations, and differences between training images and real-world conditions.

It must **not** be used as the sole basis for railway safety decisions. Suspected defects require inspection by qualified railway personnel using appropriate inspection procedures.

## 👩‍💻 Author

**Shaik Salma**

B.Tech — Computer Science and Engineering  
Specialization: Artificial Intelligence and Machine Learning  
VIT-AP University

### 🔗 Project Links

- **Live Application:** [RailGuard AI on Streamlit](https://railguard-ai-capestone.streamlit.app/)
- **GitHub Repository:** [Salma1612/RailGuard-AI](https://github.com/Salma1612/RailGuard-AI)

---

⭐ If you find this project interesting, consider starring the repository on GitHub.

**RailGuard AI — Exploring safer railway infrastructure through AI-powered image classification.**
