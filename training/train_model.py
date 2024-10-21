import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Dropout 
import json
import sys
import matplotlib.pyplot as plt

#the configuration file
import sys

if len(sys.argv) > 1:
    config_file = sys.argv[1]
else:
    config_file = "default_config.json"  # or any other default value
with open(config_file, 'r') as config:
    settings = json.load(config)

#settings from config.json
data_path = settings['data_path']
model_path = settings['model_path']
sequence_length = settings['sequence_length']
epochs = settings['epochs']
batch_size = settings['batch_size']

data = pd.read_csv(data_path)

data['saledate'] = pd.to_datetime(data['saledate'])
data = data.sort_values('saledate')

values = data['MA'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(values)

x = []
y = []
for i in range(len(scaled_values) - sequence_length):
    x.append(scaled_values[i:i + sequence_length])
    y.append(scaled_values[i + sequence_length])

x, y = np.array(x), np.array(y)

#split data into training and test sets
train_size = int(len(x) * 0.8)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

#3D  reshape for LSTM (samples, timesteps, features)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

model.save(model_path)

predicted_values = model.predict(x_test)
predicted_values = scaler.inverse_transform(predicted_values)

actual_values = scaler.inverse_transform(y_test.reshape(-1, 1))

# visualization of actual vs predicted data
plt.plot(actual_values, color='blue', label='Actual MA')
plt.plot(predicted_values, color='orange', label='Predicted MA')
plt.title('Actual vs Predicted MA (LSTM)')
plt.xlabel('Time')
plt.ylabel('Moving Average of Median Prices')
plt.legend()
plt.show()
