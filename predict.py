import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the testing dataset
dataset_test = pd.read_csv('testing_data.csv')

# Clean and Convert Data (similar to what you did for training data)
dataset_test['Close/Last'] = dataset_test['Close/Last'].replace('[\$,]', '', regex=True).astype(float)

# Extract the 'Close/Last' column for testing
testing_set = dataset_test.iloc[:, 1:2].values

sc = MinMaxScaler(feature_range=(0, 1))
testing_set_scaled = sc.fit_transform(testing_set)

sequence_length = 5

X_test = []
y_test = []

# Create sequences for testing similar to how you did for training
for i in range(sequence_length, len(testing_set_scaled)):
    X_test.append(testing_set_scaled[i-sequence_length:i, 0])
    y_test.append(testing_set_scaled[i, 0])

# Convert lists to numpy arrays
X_test, y_test = np.array(X_test), np.array(y_test)

# Reshape X_test to match the LSTM input shape
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

predicted_prices = regressor.predict(X_test)

# Inverse transform the predictions to get original scale
predicted_prices = sc.inverse_transform(predicted_prices)

# Inverse transform y_test to get original scale for actual prices
y_test_original = sc.inverse_transform(y_test.reshape(-1, 1))

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(y_test_original, color='blue', label='Actual Prices')
plt.plot(predicted_prices, color='red', label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
