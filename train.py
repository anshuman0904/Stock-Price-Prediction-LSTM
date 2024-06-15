import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dropout,Dense

import pandas as pd

file_path = 'apple.csv'

data = pd.read_csv(file_path)

# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Split the data into training and testing sets
training_data = data[data['Date'] < '2024-01-01']
testing_data = data[data['Date'] >= '2024-01-01']

# Save the datasets to separate CSV files
training_data.to_csv('training_data.csv', index=False)
testing_data.to_csv('testing_data.csv', index=False)


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the training dataset
dataset_train = pd.read_csv('training_data.csv')

# Clean and Convert Data
dataset_train['Close/Last'] = dataset_train['Close/Last'].replace('[\$,]', '', regex=True).astype(float)

# Extract the 'Close/Last' column for training
training_set = dataset_train.iloc[:, 1:2].values

# Apply MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Adjust the loop range based on the size of training_set_scaled
X_train = []
y_train = []

# Ensure the loop range is correct based on the size of training_set_scaled
for i in range(5, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-5:i, 0])
    y_train.append(training_set_scaled[i, 0])

# Convert lists to numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

print(regressor.summary())

regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

model.save('apple_stock.h5')
