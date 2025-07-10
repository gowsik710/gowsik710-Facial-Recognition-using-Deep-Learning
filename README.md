# 👤 Facial Recognition using Deep Learning

## 📦 Data Preparation

To improve the model’s ability to generalize and avoid overfitting, **data augmentation** techniques are applied:

- **Rescaling**: Pixel values of images are normalized by dividing by 255 (scaling from [0–255] to [0–1]).
- **Augmentations on Training Set**:
  - Shear (small rotations/skews)
  - Zoom in/out
  - Horizontal flipping
- These augmentations simulate different angles and lighting conditions to increase the diversity of training data.

> ✅ Note: The **testing dataset** is only rescaled (no augmentation) to ensure unbiased performance evaluation.

---

## 🏗 Model Architecture

This project uses a **Convolutional Neural Network (CNN)** designed for binary classification (e.g., face vs. non-face). The architecture includes:

### 🔹 Convolutional Layers
- **Conv Layer 1**: 32 filters, 5×5 kernel, ReLU activation
- **Conv Layer 2**: 64 filters, 5×5 kernel, ReLU activation

### 🔹 Max Pooling
- Each convolutional layer is followed by a **MaxPooling** layer to reduce spatial dimensions and overfitting.

### 🔹 Flattening
- The output from convolutional blocks is **flattened** into a 1D vector for fully connected layers.

### 🔹 Fully Connected Layers
- **Dense Layer**: 32 units, ReLU activation
- **Dropout**: 0.4 (40% neurons randomly dropped during training to reduce overfitting)

### 🔹 Output Layer
- **1 neuron**, Sigmoid activation (for binary classification)

---

## 🏃 Training Process

- **Optimizer**: Adam (adaptive learning rate for faster convergence)
- **Loss Function**: Binary Cross-Entropy (suitable for two-class classification)
- **Epochs**: 300
- **Batch Size**: 32
- **Input Image Size**: 64×64 pixels

Training is monitored using validation accuracy and validation loss.

---

## 📊 Results and Observations

- ✅ Validation accuracy improves and reaches **~70% by epoch 100**.
- ❌ After epoch 100, **accuracy declines** and **validation loss increases**.
- By epoch 300:
  - Accuracy drops to ~60%
  - Clear signs of **overfitting**

---

## 🔍 Conclusion & Recommendations

Overfitting occurred due to prolonged training and limited dataset diversity.

### 🚧 Mitigation Strategies:
- **Early Stopping**: Stop training when validation performance plateaus.
- **More Data**: Increase dataset size and variation.
- **Enhanced Augmentation**: Use more complex transformations.
- **Regularization**: Add L2 regularization or increase dropout rate.
- **Simplify Model**: Reduce complexity of the architecture to match data capacity.

---

## 🛠 Tools & Libraries Used

- **Programming Language**: Python
- **Libraries**:
  - `TensorFlow`, `Keras` – model building & training
  - `NumPy`, `pandas` – data manipulation
  - `matplotlib` – visualization

---

## 📁 Project Structure (Optional)

