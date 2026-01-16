# Stock Price Prediction with Simple Feedforward Neural Network

A from-scratch implementation of a **single-hidden-layer neural network** for predicting next-day stock closing prices using historical OHLCV data.

Compares **ReLU** vs **Tanh** activation functions, includes early stopping, min-max scaling, train/val/test split, model save/load, and nice matplotlib visualizations.

https://github.com/YOUR-USERNAME/stock-price-nn-predictor  ← replace with your repo link

## Features

- Pure NumPy neural network (no PyTorch/TensorFlow/Keras)
- Supports ReLU and Tanh activations
- Early stopping based on validation loss
- Min-Max scaling with proper inverse transformation
- Uses multiple features: Close, Open, High, Low, Volume
- Look-back window (time-series sliding window)
- GUI file picker (tkinter) to select CSV
- Saves and loads model weights (.npz format)
- Plots actual vs predicted prices + training/validation loss curves
- Computes RMSE and MAE on test set

## Demo Screenshots

*(Add 2–4 screenshots here after running the code)*

Examples:
- Training & validation loss curves
- Actual vs predicted prices (ReLU)
- Actual vs predicted prices (Tanh)

## Requirements

```text
Python 3.8+
numpy
pandas
matplotlib
tkinter     (usually comes with Python on Windows/macOS; on Linux → sudo apt install python3-tk)
