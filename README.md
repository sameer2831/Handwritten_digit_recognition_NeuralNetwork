# 🔢 Handwritten Digit Recognition using Neural Network

This project implements a **Multi-Layer Perceptron (MLP)** based Neural Network for recognizing handwritten digits using a **Kaggle Handwritten Digits Dataset** (not MNIST) on **Google Colab**.

The model is trained to classify images of digits (0–9) using fully connected layers, and it evaluates accuracy using standard performance metrics and plots.

---

## 📁 Dataset

- **Source**: train, test
- **Format**: CSV (first column is the label, the rest are pixel values)
- **Image Dimensions**: 28 x 28 (flattened to 784)

---

## 📂 Project Structure

```bash
├── Handwritten_digit_recognition.ipynb  # Main Google Colab notebook
├── dataset/
│   └── train.csv, test.csv                # CSV file with pixel values and labels
└── README.md                           # Project overview and instructions
```

---

## 🔧 Technologies Used

- Python 3.10+
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- Google Colab
- Kaggle API (for dataset download)

---

## 🚀 Getting Started

### 1. Load Notebook in Google Colab
Upload `Handwritten_digit_recognition.ipynb` to [Google Colab](https://colab.research.google.com/).

---

### 2. Dataset Download via Kaggle API

```bash
# Upload your kaggle.json file
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download and extract dataset
!kaggle datasets download -d <dataset-name>
!unzip '*.zip' -d dataset
```

### 🧠 Model Architecture (Neural Network)
A simple MLP model using Keras Sequential API:

```bash
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
  Dense(128, activation='relu', input_shape=(784,)),
  Dropout(0.2),
  Dense(64, activation='relu'),
  Dense(10, activation='softmax')  # 10 output classes for digits 0-9
])
```
- **Loss Function**: categorical_crossentropy
- **Optimizer**: Adam
- **Metrics**: accuracy

---

### ⚙️ Training Pipeline
- Data Preprocessing
 -- Normalize pixel values to [0, 1]
 -- One-hot encode digit labels
 -- Split into training and test sets
  
- Model Compilation & Training
 -- Fit the model with training data
 -- Use validation split or test set for evaluation
  
- Evaluation
 --Plot training/validation accuracy and loss

## 📌 Future Improvements

- Use CNNs for improved spatial feature extraction
- Add real-time digit drawing and prediction using OpenCV
- Deploy as a web app using Streamlit or Flask
- Implement advanced training techniques like learning rate scheduling or early stopping

---

## 👨‍💻 Author
Sameer Ashok Balkawade
MS in Computer Science, Syracuse University
