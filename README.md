# Handwritten Urdu Character Recognition System Using CNN

## 📖 Project Overview

The **Handwritten Urdu Character Recognition System Using CNN** is a deep learning based application designed to recognize handwritten Urdu characters from images. The system utilizes a Convolutional Neural Network (CNN) model trained on a multi-class Urdu character dataset and provides predictions through an interactive graphical user interface (GUI).

The objective of this project is to automate the recognition of handwritten Urdu characters and demonstrate the application of deep learning techniques in image classification tasks.

---

## 🎯 Project Objectives

- Recognize handwritten Urdu characters accurately.
- Develop a CNN-based classification model.
- Provide a user-friendly graphical interface.
- Predict characters with confidence scores.
- Evaluate model performance using accuracy and loss metrics.

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Tkinter (GUI)
- PIL (Pillow)

---

## 📂 Dataset Information

- Dataset Type: Handwritten Urdu Characters
- Total Classes: 40 Urdu Characters
- Training Images: 28,000+
- Testing Images: Multiple samples for evaluation

The dataset contains handwritten samples of different Urdu characters collected for training and testing purposes.

---

## 🧠 Model Architecture

The system uses a **Convolutional Neural Network (CNN)** consisting of:

- Rescaling Layer
- Convolutional Layers
- Max Pooling Layers
- Flatten Layer
- Dense Layers
- Dropout Layer
- Softmax Output Layer

---

## 📊 Model Performance

| Metric | Value |
|----------|----------|
| Training Accuracy | 98%+ |
| Validation Accuracy | 93.71% |
| Loss | 0.27 |

The model achieved high classification accuracy and demonstrated strong performance on unseen test samples.

---

## ✨ Features

- Image Upload Functionality
- Urdu Character Prediction
- Confidence Score Display
- CNN-Based Recognition
- Modern GUI Dashboard
- Training & Evaluation Support
- Accuracy Graph
- Loss Graph
- Prediction History

---

## 🖥 GUI Dashboard

The application provides an interactive dashboard where users can:

- Upload handwritten Urdu character images.
- Preview uploaded images.
- Predict character labels.
- View prediction confidence scores.
- Access prediction history.

---

## 📁 Project Structure

```text
Handwritten_Urdu_Character_Recognition_System_Using_CNN
│
├── Dataset
│   ├── characters_train_set
│   └── characters_test_set
│
├── Gui
│   └── gui.py
│
├── Models
│   ├── urdu_model.keras
│   └── class_names.txt
│
├── Results
│   ├── accuracy_graph.png
│   └── loss_graph.png
│
├── Screenshots
│
├── Source_Code
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── preprocessing.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python Source_Code/train.py
```

### 3. Evaluate the Model

```bash
python Source_Code/evaluate.py
```

### 4. Run Prediction

```bash
python Source_Code/predict.py
```

### 5. Launch GUI

```bash
python Gui/gui.py
```

---

## 📈 Results

The trained CNN model successfully recognizes handwritten Urdu characters with high accuracy and provides predictions through a graphical user interface.

Performance graphs are available in the **Results** folder.

---

## 🔮 Future Enhancements

- Real-time handwriting recognition.
- Urdu word recognition.
- Mobile application integration.
- Web-based deployment.
- Advanced CNN architectures.

---

## 👨‍💻 Developer

**Sameer Ahmed**

Student | BS Data Science

---

## 📌 Project Title

**Handwritten Urdu Character Recognition System Using CNN**
