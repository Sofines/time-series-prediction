import numpy as np # type: ignore
import pandas as pd # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import json
import sys

#the configuration file
config_file = sys.argv[1]
with open(config_file, 'r') as config:
    settings = json.load(config)

#the settings
data_path = settings['data_path']
model_path = settings['model_path']
sequence_length = settings['sequence_length']
results_folder = settings['results_folder']

if not results_folder.endswith('/'):
    results_folder += '/'

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

# Convert lists to numpy arrays
x = np.array(x)
y = np.array(y)

# Split data into training and test sets (only test set is needed for evaluation)
train_size = int(len(x) * 0.8)
x_test, y_test = x[train_size:], y[train_size:]

# Reshape input for LSTM: (samples, timesteps, features)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Load the pre-trained model
model = load_model(model_path)

# Make predictions on the test set
predicted_values = model.predict(x_test)

# Inverse transform the predictions back to the original scale
predicted_values = scaler.inverse_transform(predicted_values)

# Inverse transform the actual values back to the original scale
actual_values = scaler.inverse_transform(y_test.reshape(-1, 1))

# Save predictions to predictions.csv in the results folder
predictions_df = pd.DataFrame({
    'Actual MA': actual_values.flatten(),
    'Predicted MA': predicted_values.flatten()
})

predictions_df.to_csv(results_folder + 'predictions.csv', index=False)
print("Predictions saved to results/predictions.csv")

# Evaluate metrics (you can import or define your metrics utility)
from metrics_utils import evaluate_metrics
metrics = evaluate_metrics(actual_values, predicted_values)

for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")

metrics_dict = {metric_name: metric_value for metric_name, metric_value in metrics.items()}

# Save metrics to evaluation_metrics.json
with open(results_folder + 'evaluation_metrics.json', 'w') as json_file:
    json.dump(metrics_dict, json_file, indent=4)
print("Metrics saved to results/evaluation_metrics.json")

# Visualization
import matplotlib.pyplot as plt # type: ignore
plt.figure(figsize=(10, 6))
plt.plot(actual_values, color='blue', label='Actual MA')
plt.plot(predicted_values, color='orange', label='Predicted MA')
plt.title('Actual vs Predicted MA (Evaluation)')
plt.xlabel('Time')
plt.ylabel('Moving Average of Median Prices')
plt.legend()
plt.show()
