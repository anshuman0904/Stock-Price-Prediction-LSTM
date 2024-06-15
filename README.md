# Stock Price Prediction using LSTM

This repository contains a deep learning model implemented in TensorFlow/Keras for predicting stock prices based on historical data. The model uses Long Short-Term Memory (LSTM) networks, a type of recurrent neural network (RNN) well-suited for sequence prediction tasks.

## Overview

The LSTM model is trained on historical stock price data to learn patterns and trends. Once trained, it can predict future stock prices based on the input data provided. This repository includes scripts for data preprocessing, model training, and prediction.

## Usage

### Requirements

- Python 3.x
- TensorFlow 2.x
- pandas
- numpy
- matplotlib

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/Stock-Price-Prediction-LSTM.git
   cd Stock-Price-Prediction-LSTM
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
    ```
### Training

1. Prepare your data
   * Ensure your training data is in CSV format with columns 'Date', 'Close/Last', 'Volume', 'Open', 'High', 'Low'.
2. Run the training script:
    ```bash
    python train.py --data_path path_to_training_data.csv
    ```
3. The trained model will be saved after training.

### Prediction

1. Prepare your testing data:
   * Ensure your testing data is in CSV format similar to training data.
2. Run the prediction script:
   ```bash
   python predict.py --data_path path_to_testing_data.csv
    ```
3. Predicted and actual stock prices will be plotted for visualization.
