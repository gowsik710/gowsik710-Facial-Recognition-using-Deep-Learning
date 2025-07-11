# 🧠 Facial Recognition using CNN

**Author:** Gowsik Arivazhagan  
**Type:** Academic Project | Deep Learning | Computer Vision  
**Frameworks:** TensorFlow, Keras, NumPy  
**Dataset:** ORL Face Dataset  
**Accuracy:** ~93% Validation Accuracy

---

## 📌 Overview

This project implements a facial recognition system using a Convolutional Neural Network (CNN) on the ORL Face Dataset. The objective is to classify grayscale facial images into one of 40 individual classes. It demonstrates how deep learning can be applied to facial image data and showcases the effectiveness of CNNs for image classification tasks on small datasets.

---

## 📚 Dataset Description

The [ORL Faces Dataset](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html) is widely used in academic facial recognition research. It contains:

- **Total Images:** 400 grayscale images  
- **Subjects:** 40 individuals  
- **Images per Subject:** 10  
- **Image Size:** 112 x 92 pixels  
- **Format Used:** `.npz` (NumPy compressed format)  
- **Data Split:**
  - Training: 240 images
  - Testing: 160 images

Each image is labeled from 0 to 39, corresponding to each unique individual.

---

## ⚙️ Tools & Libraries Used

- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- Jupyter Notebook

---

## 🏗️ Project Pipeline

### 1. Data Loading
- Loaded the `.npz` file containing pre-split data.
- Extracted `x_train`, `x_test`, `y_train`, `y_test`.

### 2. Preprocessing
- Images reshaped to `(batch_size, 112, 92, 1)` to match CNN input format.
- Pixel values normalized between 0 and 1.
- Labels one-hot encoded for multi-class classification.

### 3. Model Architecture

The model is a simple yet effective Convolutional Neural Network:

- **Conv2D Layer:** 32 filters, (3x3) kernel, ReLU activation  
- **MaxPooling2D:** Pool size (2x2)  
- **Flatten Layer**  
- **Dense Layer:** 256 neurons with ReLU  
- **Dropout Layer:** 0.5 rate to prevent overfitting  
- **Output Layer:** 40 neurons with softmax (for 40 classes)

### 4. Model Compilation

- **Loss Function:** Categorical Crossentropy  
- **Optimizer:** Adam  
- **Metrics:** Accuracy  
- **Epochs:** 99  
- **Batch Size:** 128  
- **Validation:** Evaluated using test set

### 5. Model Training

- Trained on 240 training images
- Tested on 160 validation images
- Achieved consistent improvement across epochs
- Minimal overfitting due to dropout regularization

---

## 📊 Model Performance

The model achieved:

- **~93% validation accuracy** by the end of training
- Smooth convergence with increasing accuracy and decreasing loss
- Strong generalization on unseen data

### Training Snapshot:

| Epoch Range | Train Accuracy | Validation Accuracy |
|-------------|----------------|----------------------|
| 0–10        | ~20%           | ~68%                 |
| 10–20       | ~61%           | ~85%                 |
| 20–30       | ~91%           | ~92%                 |
| 30–99       | ~96%           | ~93%                 |

---

## ✅ Results & Observations

- CNN successfully learned facial features with limited training data
- Dropout layer helped control overfitting
- Achieved high classification accuracy with a basic architecture
- Suitable for small-scale facial recognition problems

---

## 💡 Key Takeaways

- Deep learning models like CNNs perform well even on small grayscale datasets
- Proper preprocessing (reshaping, normalization) is critical
- Simple models can be powerful with appropriate tuning
- Validation accuracy is a good proxy for real-world performance

---

## 🚀 Future Work

Here are possible improvements for future versions of this project:

- ✅ **Data Augmentation:** Apply rotation, flipping, scaling to improve generalization  
- ✅ **Transfer Learning:** Use pretrained models like MobileNetV2 or VGGFace  
- ✅ **Face Detection Preprocessing:** Integrate OpenCV or Dlib for automatic face cropping  
- ✅ **Evaluation Metrics:** Add confusion matrix, per-class precision, recall, F1-score  
- ✅ **Deployment:** Convert model to TensorFlow Lite for mobile deployment or create a simple web app using Streamlit or Flask

