# Pneumonia Detection Using CNN

This repository contains code for training and evaluating a Convolutional Neural Network (CNN) for detecting pneumonia from chest X-ray images. The project leverages TensorFlow and Keras to build, train, and evaluate the deep learning model, with metrics and charts to visualize performance.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Model Architecture](#model-architecture)
6. [Training and Evaluation](#training-and-evaluation)
7. [Results](#results)
8. [How to Use](#how-to-use)
9. [Contributing](#contributing)
10. [License](#license)

---

## Project Overview
This project aims to detect pneumonia from chest X-ray images using a CNN-based model. It includes:
- Data preprocessing and augmentation.
- Model training, validation, and testing.
- Evaluation metrics: Accuracy, Precision, Recall, Specificity, and F1 Score.
- Visualization of training/validation loss, AUC, confusion matrix, and ROC curve.

---

## Dataset
The dataset used for this project is the "Chest X-Ray Images (Pneumonia)" dataset, which consists of:
- **Train Dataset**: For training the model.
- **Validation Dataset**: For tuning hyperparameters.
- **Test Dataset**: For final model evaluation.

### Dataset Directory Structure
```
chest_xray/
    train/
    val/
    test/
```
- Each directory contains subfolders for classes (e.g., `NORMAL` and `PNEUMONIA`).

---

## Installation

### Prerequisites
Ensure the following are installed:
- Python 3.7+
- Google Colab or local Jupyter Notebook

### Required Libraries
Install the required Python libraries:
```bash
pip install tensorflow keras matplotlib numpy pandas seaborn
```

---

## Project Structure
```
.
|-- chest_xray/                # Dataset folder
|-- cnn_pneumonia.ipynb        # Main notebook containing the code
|-- README.md                  # Documentation
|-- results/                   # Output charts and results
```

---

## Model Architecture
The CNN architecture consists of:
1. **Input Layer**: Accepts input images of shape (64, 64, 1).
2. **Convolutional Layers**: Extract features using 3 convolutional layers with ReLU activation.
3. **Pooling Layers**: Reduce spatial dimensions using max-pooling layers.
4. **Flatten Layer**: Converts 2D feature maps to 1D feature vectors.
5. **Dense Layers**: Fully connected layers with ReLU and sigmoid activations.

### Hyperparameters
- Image Dimension: `64x64`
- Batch Size: `128`
- Epochs: `100`
- Optimizer: `Adam`
- Loss Function: `Binary Crossentropy`

---

## Training and Evaluation

### Training
The model is trained using:
- **Data Augmentation**: Includes rescaling, shear, zoom, and horizontal flipping.
- **Early Stopping**: Stops training when validation performance stops improving.

### Evaluation Metrics
1. **Accuracy**
2. **Precision**
3. **Recall**
4. **Specificity**
5. **F1 Score**

### Charts
1. Training vs. Validation Loss
2. Training vs. Validation AUC
3. Confusion Matrix
4. ROC Curve

---

## Results
The model achieved the following metrics on the test dataset:
- **Accuracy**: 91.99%
- **Precision**: 90.09%
- **Recall**: 97.95%
- **Specificity**: 82.05%
- **F1 Score**: 93.86%

---

## How to Use

### Step 1: Set Up the Dataset
Upload the dataset to Google Drive and set the appropriate paths in the code:
```python
project_path = "/content/drive/MyDrive/chest_xray"
```

### Step 2: Run the Notebook
Execute the notebook sequentially to train the model and generate metrics and charts.

### Step 3: Visualize Results
Generated charts and confusion matrices are displayed within the notebook.

---

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with clear documentation of your changes.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

