import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

class NeuralNetwork:
    """A simple feedforward neural network for stock price prediction."""
    
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        """
        Initialize the neural network with given architecture and activation function.
        
        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Number of output neurons (1 for regression).
            activation (str): Activation function for hidden layer ('relu' or 'tanh').
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def _relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        """Derivative of ReLU activation function."""
        return np.where(x > 0, 1, 0)

    def _tanh(self, x):
        """Tanh activation function."""
        return np.tanh(x)

    def _tanh_derivative(self, x):
        """Derivative of tanh activation function."""
        return 1 - np.tanh(x) ** 2

    def _sigmoid(self, x):
        """Sigmoid activation function with clipping to prevent overflow."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _sigmoid_derivative(self, x):
        """Derivative of sigmoid activation function."""
        return x * (1 - x)

    def feedforward(self, X):
        """
        Perform forward propagation through the network.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, input_size).
        
        Returns:
            np.ndarray: Predicted output of shape (n_samples, output_size).
        """
        self.hidden_activation = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        if self.activation == 'relu':
            self.hidden_output = self._relu(self.hidden_activation)
        else:  # tanh
            self.hidden_output = self._tanh(self.hidden_activation)
        self.output_activation = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = self._sigmoid(self.output_activation)
        return self.predicted_output

    def backward(self, X, y, learning_rate):
        """
        Perform backpropagation to update weights and biases.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, input_size).
            y (np.ndarray): Target data of shape (n_samples, output_size).
            learning_rate (float): Learning rate for gradient descent.
        """
        output_error = y - self.predicted_output
        output_delta = output_error * self._sigmoid_derivative(self.predicted_output)

        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        if self.activation == 'relu':
            hidden_delta = hidden_error * self._relu_derivative(self.hidden_output)
        else:  # tanh
            hidden_delta = hidden_error * self._tanh_derivative(self.hidden_output)

        self.weights_hidden_output += learning_rate * np.dot(self.hidden_output.T, output_delta)
        self.bias_output += learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        self.weights_input_hidden += learning_rate * np.dot(X.T, hidden_delta)
        self.bias_hidden += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

    def train(self, X_train, y_train, X_val, y_val, epochs, learning_rate, patience=100):
        """
        Train the neural network with early stopping.
        
        Args:
            X_train (np.ndarray): Training input data.
            y_train (np.ndarray): Training target data.
            X_val (np.ndarray): Validation input data.
            y_val (np.ndarray): Validation target data.
            epochs (int): Maximum number of training epochs.
            learning_rate (float): Learning rate for gradient descent.
            patience (int): Number of epochs to wait before stopping if no improvement.
        
        Returns:
            tuple: Lists of training and validation losses.
        """
        train_loss_list = []
        val_loss_list = []
        best_val_loss = float('inf')
        best_epoch = 0
        best_weights = None

        for epoch in range(epochs):
            # Training
            train_output = self.feedforward(X_train)
            self.backward(X_train, y_train, learning_rate)
            train_loss = np.mean(np.square(y_train - train_output))
            train_loss_list.append(train_loss)

            # Validation
            val_output = self.feedforward(X_val)
            val_loss = np.mean(np.square(y_val - val_output))
            val_loss_list.append(val_loss)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_weights = {
                    'weights_input_hidden': self.weights_input_hidden.copy(),
                    'weights_hidden_output': self.weights_hidden_output.copy(),
                    'bias_hidden': self.bias_hidden.copy(),
                    'bias_output': self.bias_output.copy()
                }
            elif epoch - best_epoch >= patience:
                print(f"Early stopping at epoch {epoch}. Best validation loss: {best_val_loss:.6f}")
                # Restore best weights
                self.weights_input_hidden = best_weights['weights_input_hidden']
                self.weights_hidden_output = best_weights['weights_hidden_output']
                self.bias_hidden = best_weights['bias_hidden']
                self.bias_output = best_weights['bias_output']
                break

        return train_loss_list, val_loss_list

    def predict(self, X):
        """
        Generate predictions for input data.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, input_size).
        
        Returns:
            np.ndarray: Predicted output of shape (n_samples, output_size).
        """
        return self.feedforward(X)

    def save_model(self, file_path):
        """
        Save model weights and biases to a file.
        
        Args:
            file_path (str): Path to save the model (.npz file).
        """
        np.savez(file_path, 
                 weights_input_hidden=self.weights_input_hidden,
                 weights_hidden_output=self.weights_hidden_output,
                 bias_hidden=self.bias_hidden,
                 bias_output=self.bias_output,
                 activation=self.activation)

    def load_model(self, file_path):
        """
        Load model weights and biases from a file.
        
        Args:
            file_path (str): Path to the model file (.npz file).
        """
        try:
            data = np.load(file_path)
            self.weights_input_hidden = data['weights_input_hidden']
            self.weights_hidden_output = data['weights_hidden_output']
            self.bias_hidden = data['bias_hidden']
            self.bias_output = data['bias_output']
            self.activation = str(data['activation'])
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")

def min_max_scaler(data):
    """
    Scale data to [0, 1] range using MinMaxScaler.
    
    Args:
        data (np.ndarray): Input data to scale.
    
    Returns:
        tuple: Scaled data, minimum value, maximum value.
    """
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    if np.any(max_val == min_val):
        raise ValueError("Data has no variation in one or more features (min = max).")
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data, min_val, max_val

def inverse_min_max_scaler(data, min_val, max_val):
    """
    Inverse transform scaled data back to original scale.
    
    Args:
        data (np.ndarray): Scaled data.
        min_val (np.ndarray): Minimum values used for scaling.
        max_val (np.ndarray): Maximum values used for scaling.
    
    Returns:
        np.ndarray: Data in original scale.
    """
    return data * (max_val - min_val) + min_val

def prepare_data(stock_data, look_back, target_col=0):
    """
    Prepare data for neural network with look-back window and multiple features.
    
    Args:
        stock_data (np.ndarray): Scaled stock data with multiple features.
        look_back (int): Number of past time steps to use as input.
        target_col (int): Index of the target column (e.g., 0 for Close).
    
    Returns:
        tuple: Input features (X) and target values (y).
    """
    if len(stock_data) < look_back + 1:
        raise ValueError(f"Dataset too small for look_back={look_back}. Need at least {look_back + 1} data points.")
    X, y = [], []
    for i in range(len(stock_data) - look_back):
        X.append(stock_data[i:i + look_back].flatten())  # Flatten look_back * n_features
        y.append(stock_data[i + look_back, target_col])
    return np.array(X), np.array(y).reshape(-1, 1)

def train_test_split_manual(X, y, test_size=0.2, val_size=0.2):
    """
    Split data into training, validation, and test sets.
    
    Args:
        X (np.ndarray): Input features.
        y (np.ndarray): Target values.
        test_size (float): Proportion of data for test set.
        val_size (float): Proportion of data for validation set.
    
    Returns:
        tuple: Training, validation, and test sets (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    n = len(X)
    test_split = int(n * (1 - test_size))
    val_split = int(n * (1 - test_size - val_size))
    if val_split < 1 or test_split - val_split < 1 or n - test_split < 1:
        raise ValueError("Dataset too small for train-val-test split.")
    X_train, X_val, X_test = X[:val_split], X[val_split:test_split], X[test_split:]
    y_train, y_val, y_test = y[:val_split], y[val_split:test_split], y[test_split:]
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_stock_data(file_path, features=['Close', 'Open', 'High', 'Low', 'Volume']):
    """
    Load and preprocess stock data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        features (list): List of feature columns to include.
    
    Returns:
        tuple: Scaled stock data, min values, max values.
    """
    try:
        df = pd.read_csv(file_path)
        missing_cols = [col for col in features if col not in df.columns]
        if missing_cols:
            print(f"Warning: Columns {missing_cols} not found in CSV. Using available columns.")
            features = [col for col in features if col in df.columns]
        if not features:
            raise ValueError("No valid feature columns found in CSV.")
        df = df[features]
        stock_data, min_val, max_val = min_max_scaler(df.values)
        return stock_data, min_val, max_val
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please check the file path.")
        return None, None, None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None, None

def compute_metrics(actual, predicted):
    """
    Compute RMSE and MAE metrics for predictions.
    
    Args:
        actual (np.ndarray): Actual values.
        predicted (np.ndarray): Predicted values.
    
    Returns:
        tuple: RMSE and MAE values.
    """
    rmse = np.sqrt(np.mean(np.square(actual - predicted)))
    mae = np.mean(np.abs(actual - predicted))
    return rmse, mae

def plot_predictions(actual_prices, predicted_prices, min_val, max_val, title="Stock Price Prediction"):
    """
    Plot actual vs predicted stock prices.
    
    Args:
        actual_prices (np.ndarray): Actual target values.
        predicted_prices (np.ndarray): Predicted target values.
        min_val (float): Minimum value for inverse scaling (Close price).
        max_val (float): Maximum value for inverse scaling (Close price).
        title (str): Plot title.
    """
    actual_prices = inverse_min_max_scaler(actual_prices, min_val, max_val).flatten()
    predicted_prices = inverse_min_max_scaler(predicted_prices, min_val, max_val).flatten()

    min_len = min(len(actual_prices), len(predicted_prices))
    actual_prices = actual_prices[:min_len]
    predicted_prices = predicted_prices[:min_len]

    plt.figure(figsize=(12, 7))
    plt.style.use('ggplot')
    plt.plot(actual_prices, label='Actual Prices', color='green', marker='o', linestyle='--', markersize=6, linewidth=2)
    plt.plot(predicted_prices, label='Predicted Prices', color='red', marker='x', linestyle='-', markersize=6, linewidth=2)
    plt.title(title, fontsize=18, fontweight='bold', color='blue')
    plt.xlabel("Time (Days)", fontsize=14, color='darkblue')
    plt.ylabel("Stock Price", fontsize=14, color='darkblue')
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.xticks(np.arange(0, min_len, step=max(1, min_len//10)), fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_loss(train_loss, val_loss, title="Training and Validation Loss"):
    """
    Plot training and validation loss over epochs.
    
    Args:
        train_loss (list): Training loss values.
        val_loss (list): Validation loss values.
        title (str): Plot title.
    """
    plt.figure(figsize=(10, 6))
    plt.style.use('ggplot')
    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='orange')
    plt.title(title, fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()

def select_file_gui():
    """
    Open a GUI file dialog for selecting a CSV file.
    
    Returns:
        str: Selected file path or empty string if cancelled.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select Stock Data CSV File",
        filetypes=[("CSV Files", "*.csv")]
    )
    root.destroy()
    return file_path

def main():
    """Main function to run the stock price prediction pipeline."""
    # Select file via GUI
    file_path = select_file_gui()
    if not file_path:
        print("No file selected. Exiting.")
        return
    file_path = file_path.replace('\\', '/')

    # Load data
    features = ['Close', 'Open', 'High', 'Low', 'Volume']
    stock_data, min_val, max_val = load_stock_data(file_path, features)
    if stock_data is None:
        return

    # Parameters
    look_back = 10
    epochs = 10000
    learning_rate = 0.001
    hidden_size = 20
    patience = 100
    input_size = look_back * stock_data.shape[1]  # look_back * n_features
    output_size = 1
    target_col = features.index('Close') if 'Close' in features else 0

    try:
        # Prepare data
        X, y = prepare_data(stock_data, look_back, target_col)
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_manual(X, y, test_size=0.2, val_size=0.2)

        # Train models with different activations
        for activation in ['relu', 'tanh']:
            print(f"\nTraining with {activation.upper()} activation")
            nn = NeuralNetwork(input_size, hidden_size, output_size, activation=activation)
            train_loss, val_loss = nn.train(X_train, y_train, X_val, y_val, epochs, learning_rate, patience)

            # Generate predictions
            predictions = nn.predict(X_test)
            rmse, mae = compute_metrics(y_test, predictions)
            print(f"{activation.upper()} - Test RMSE: {rmse:.6f}, Test MAE: {mae:.6f}")

            # Plot results
            plot_predictions(y_test, predictions, min_val[target_col], max_val[target_col], 
                           title=f"Stock Price Prediction ({activation.upper()} Activation)")
            plot_loss(train_loss, val_loss, title=f"Loss Curves ({activation.upper()} Activation)")

            # Save model
            model_path = f"nn_model_{activation}.npz"
            nn.save_model(model_path)
            print(f"Model saved to {model_path}")

            # Test loading model
            nn_loaded = NeuralNetwork(input_size, hidden_size, output_size, activation=activation)
            nn_loaded.load_model(model_path)
            loaded_predictions = nn_loaded.predict(X_test)
            loaded_rmse, loaded_mae = compute_metrics(y_test, loaded_predictions)
            print(f"Loaded Model ({activation.upper()}) - Test RMSE: {loaded_rmse:.6f}, Test MAE: {loaded_mae:.6f}")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()