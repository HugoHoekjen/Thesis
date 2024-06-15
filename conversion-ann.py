import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from keras.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
import random
import matplotlib.dates as mdates

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Data Preprocessing
df = pd.read_csv('conversion_rate_2_years.csv')

# Parse the 'date' column as datetime and set it as the index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Extract the 'conversion_rate' column
conversion_rate = df['conversion_rate'].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(conversion_rate)

# Split data into train and test sets
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# Define window size
window_size = 8

# Prepare data for training
def prepare_data(data, window_size):
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        Y.append(data[i + window_size])
    return np.array(X), np.array(Y)

train_X, train_Y = prepare_data(train_data, window_size)
test_X, test_Y = prepare_data(test_data, window_size)

# Define and train model
def build_model(window_size):
    model = Sequential()
    model.add(Dense(8, input_dim=window_size, kernel_initializer="uniform", activation="sigmoid"))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=Adam())
    return model

# Implement time series cross-validation on the training set
tscv = TimeSeriesSplit(n_splits=5)

# mae_scores = []
# rmse_scores = []
# mape_scores = []
#
# for train_index, val_index in tscv.split(train_X):
#     split_train_X, val_X = train_X[train_index], train_X[val_index]
#     split_train_Y, val_Y = train_Y[train_index], train_Y[val_index]
#
#     # Build and train the model
#     model = build_model(window_size)
#     history = model.fit(split_train_X, split_train_Y, epochs=1000, batch_size=64, validation_data=(val_X, val_Y), verbose=0)
#
#     # Make predictions
#     predictions = model.predict(val_X)
#
#     # Inverse scaling
#     predictions_rescaled = scaler.inverse_transform(predictions)
#     val_Y_rescaled = scaler.inverse_transform(val_Y.reshape(-1, 1))
#
#     # Calculate mean absolute error
#     mae = mean_absolute_error(val_Y_rescaled, predictions_rescaled)
#     mae_scores.append(mae)
#
#     # Calculate root mean squared error
#     rmse = np.sqrt(mean_squared_error(val_Y_rescaled, predictions_rescaled))
#     rmse_scores.append(rmse)
#
#     # Calculate mean absolute percentage error
#     mape = mean_absolute_percentage_error(val_Y_rescaled, predictions_rescaled)
#     mape_scores.append(mape)

# # Print the cross-validation results
# print(f'Mean MAE: {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}')
# print(f'Mean RMSE: {np.mean(rmse_scores):.3f} ± {np.std(rmse_scores):.3f}')
# print(f'Mean MAPE: {np.mean(mape_scores):.3f} ± {np.std(mape_scores):.3f}')

# Train final model on the entire training set
final_model = build_model(window_size)
final_model.fit(train_X, train_Y, epochs=1000, batch_size=64)

# Make predictions on the test set
test_predictions = final_model.predict(test_X)

# Inverse scaling
test_predictions_rescaled = scaler.inverse_transform(test_predictions)
test_Y_rescaled = scaler.inverse_transform(test_Y.reshape(-1, 1))

# Calculate mean absolute error, RMSE, and MAPE on the test set
test_mae = mean_absolute_error(test_Y_rescaled, test_predictions_rescaled)
test_rmse = np.sqrt(mean_squared_error(test_Y_rescaled, test_predictions_rescaled))
test_mape = mean_absolute_percentage_error(test_Y_rescaled, test_predictions_rescaled)

print('Test MAE: %.10f' % test_mae)
print('Test RMSE: %.10f' % test_rmse)
print('Test MAPE: %.10f' % test_mape)

# Plot actual vs predicted values for the test set with actual dates
test_dates = df.index[train_size + window_size:]

plt.plot(test_dates, test_Y_rescaled, label='Actual')
plt.plot(test_dates, test_predictions_rescaled, label='Predicted', color='red')
plt.xlabel('Date')
plt.ylabel('Conversion Rate')
plt.legend()
plt.title('Conversion Rate Predictions - ANN Model')

# Format the x-axis to show only the month
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

plt.gcf().autofmt_xdate()  # Auto-format the x-axis labels for better readability
plt.show()
