# Feed-Forward Neural Network Implementation from Scratch

This repository contains my implementation of a **Feed-Forward Neural Network (FNN)** from scratch as part of a machine learning assignment. 
The neural network was designed to classify the **FashionMNIST dataset** without using any deep learning libraries such as TensorFlow or PyTorch.

## Project Overview

The project involves building a modular and efficient FNN that supports the following features:
- **Dense Layers**: Fully connected layers.
- **Activation Functions**: ReLU for intermediate layers and Softmax for multi-class classification.
- **Optimization**: Adam optimizer for weight updates.
- **Regularization**: Dropout to prevent overfitting.
- **Testing and Evaluation**: Support for loading saved model weights and predicting labels for unseen data.

### Key Components Implemented
- **Dense Layer**: Custom implementation of fully connected layers.
- **Activation Functions**: ReLU and Softmax.
- **Dropout Regularization**: Applied during training to prevent overfitting.
- **Loss Function**: Cross-entropy loss for classification tasks.
- **Optimizer**: Adaptive Moment Estimation (Adam).
- **Backpropagation**: Manual implementation for updating weights.

---

## Dataset

The **FashionMNIST** dataset was used for training and evaluation. It consists of:
- 60,000 grayscale training images and 10,000 testing images.
- Images are 28x28 pixels with labels corresponding to 10 different classes:
  1. T-shirt/top
  2. Trouser
  3. Pullover
  4. Dress
  5. Coat
  6. Sandal
  7. Shirt
  8. Sneaker
  9. Bag
  10. Ankle boot

---

## Features
- **No External Deep Learning Libraries**: Implementation is done from scratch using only allowed libraries (e.g., NumPy, Matplotlib).
- **Modular Code**: Easily adaptable to different architectures or datasets.
- **Pickle Support**: Save and load the trained model for future predictions.

---

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/neural_network.git
   cd neural_network
2.Ensure the following files are present in the repository:

- 1905092.ipynb: Jupyter notebook containing the implementation.
- model_1905092.pickle: Pickle file for the saved model or weights.
- report_1905092.pdf: PDF report with training and evaluation results.
- fashion-mnist_train.csv: Training dataset.
- fashion-mnist_test.csv: Test dataset.
All these files must be placed in the same folder for the project to run correctly.
