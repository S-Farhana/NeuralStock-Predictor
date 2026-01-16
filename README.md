Stock Price Prediction Using Neural Network

This project implements a feedforward neural network from scratch in Python to predict stock prices using historical data. It demonstrates essential concepts in time-series forecasting, neural network design, and data preprocessing, making it a strong example for a Data Scientist portfolio.

Project Highlights

Custom Feedforward Neural Network with one hidden layer

Supports multiple activation functions: ReLU and Tanh

Early stopping to prevent overfitting and improve generalization

Handles multiple stock features: Close, Open, High, Low, Volume

Min-max normalization for data preprocessing

Look-back window approach for time-series prediction

Manual train-validation-test split

Compute performance metrics: RMSE and MAE

Save and load trained models as .npz files

Interactive file selection using Tkinter GUI

Visualization of predictions and loss curves for analysis

Requirements

Python 3.8+

Libraries: numpy, pandas, matplotlib, tkinter (usually included with Python)

Install missing packages:

pip install numpy pandas matplotlib

How to Use

Clone the repository:

git clone <your-repo-url>
cd <repo-folder>


Run the script:

python stock_price_nn.py


Select a CSV file containing historical stock data using the GUI.

The script will automatically:

Preprocess the data

Train the neural network with ReLU and Tanh activations

Generate predictions

Plot actual vs predicted stock prices and loss curves

Save trained models as .npz files

CSV File Format

The CSV file should contain the following columns:

Close (target variable), Open, High, Low, Volume

Example:

Date	Open	High	Low	Close	Volume
2023-01-01	100	105	98	102	1500000
2023-01-02	102	108	101	107	1200000
Parameters

look_back: Number of past days used for prediction (default: 10)

epochs: Maximum training epochs (default: 10000)

learning_rate: Learning rate (default: 0.001)

hidden_size: Number of neurons in the hidden layer (default: 20)

patience: Early stopping patience (default: 100)

Output

Plots:

Actual vs predicted stock prices

Training and validation loss curves

Saved Models:

nn_model_relu.npz

nn_model_tanh.npz

Performance Metrics printed to console: RMSE and MAE

Key Functions

NeuralNetwork: Feedforward neural network class

min_max_scaler / inverse_min_max_scaler: Scale and inverse-scale data

prepare_data: Prepare input-output sequences with look-back window

train_test_split_manual: Train-validation-test split

load_stock_data: Load CSV data and scale features

compute_metrics: Compute RMSE and MAE

plot_predictions / plot_loss: Visualize results

select_file_gui: GUI-based file selection

Why This Project is Important for a Data Scientist Portfolio

Demonstrates ability to implement neural networks from scratch

Shows time-series prediction skills applied to stock market data

Includes data preprocessing, visualization, and model evaluation

Highlights Python programming and problem-solving skills

License
