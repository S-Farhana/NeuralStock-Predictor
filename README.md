# Stock Price Prediction Using Neural Network

This project implements a **stock price prediction system using a feedforward neural network built from scratch in Python**.  
It focuses on understanding **time-series forecasting, neural network fundamentals, and data preprocessing** without relying on high-level machine learning libraries.

---

## Project Overview

The objective of this project is to predict future stock prices based on historical market data.  
A custom neural network is trained using past price information and evaluated using standard regression metrics.

This project is suitable for:
- Data Science portfolios  
- Machine Learning fundamentals practice  
- Academic and mini-project submissions  

---

## Key Highlights

- Feedforward Neural Network implemented from scratch  
- No use of TensorFlow, Keras, or Scikit-learn  
- Supports ReLU and Tanh activation functions  
- Time-series prediction using look-back window approach  
- Min-Max normalization for feature scaling  
- Early stopping to reduce overfitting  
- Manual train, validation, and test split  
- Performance evaluation using RMSE and MAE  
- Model saving and loading using `.npz` files  
- GUI-based CSV file selection using Tkinter  
- Visualization of predictions and loss curves  

---

## Technologies Used

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Tkinter  

---

## Dataset Description

The input dataset must be a CSV file containing historical stock market data.

Required columns:
- Open  
- High  
- Low  
- Close  
- Volume  

The **Close price** is used as the target variable.

---

## How the Model Works

1. Stock data is loaded from a CSV file  
2. Data is normalized using Min-Max scaling  
3. Input sequences are created using a look-back window  
4. Data is split into training, validation, and testing sets  
5. A neural network is trained using backpropagation  
6. Early stopping monitors validation loss  
7. Predictions are generated on test data  
8. Results are visualized and evaluated  

---

## Model Parameters

- Look-back window: 10  
- Hidden layer neurons: 20  
- Learning rate: 0.001  
- Maximum epochs: 10000  
- Early stopping patience: 100  

These parameters can be modified inside the code.

---

## Output

- Actual vs Predicted stock price plots  
- Training and validation loss curves  
- Saved trained models:
  - `nn_model_relu.npz`
  - `nn_model_tanh.npz`
- Printed evaluation metrics:
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)

---
## Installation

Install the required dependencies:

```bash
pip install numpy pandas matplotlib

---

## How to Run

Run the Python script:

```bash
python stock_price_nn.py



