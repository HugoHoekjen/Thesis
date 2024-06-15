import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from keras.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
import random
import matplotlib.dates as mdates

# Set seeds for reproducibility
np.random.seed(24)
random.seed(24)
tf.random.set_seed(24)

# Data Preprocessing
df = pd.read_csv('add_to_cart_2_years.csv')

# Parse the 'date' column as datetime and set it as the index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Extract the 'add_to_cart' column
add_to_cart = df['add_to_cart'].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(add_to_cart)

# Split data into train and test sets
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# Define window size
window_size = 10

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

# # Implement time series cross-validation on the training set -- commented out for speed as unused after hyperparameter tuning
# tscv = TimeSeriesSplit(n_splits=5)
#
# mse_scores = []
# mae_scores = []
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
#     # Calculate mean squared error and mean absolute error
#     mse = mean_squared_error(val_Y_rescaled, predictions_rescaled)
#     mae = mean_absolute_error(val_Y_rescaled, predictions_rescaled)
#
#     mse_scores.append(mse)
#     mae_scores.append(mae)
#
# # Print the cross-validation results
# print(f'Mean MSE: {np.mean(mse_scores):.3f} ± {np.std(mse_scores):.3f}')
# print(f'Mean MAE: {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}')

# Train final model on the entire training set
final_model = build_model(window_size)
final_model.fit(train_X, train_Y, epochs=500, batch_size=32)

# Make predictions on the test set
test_predictions = final_model.predict(test_X)

# Inverse scaling
test_predictions_rescaled = scaler.inverse_transform(test_predictions)
test_Y_rescaled = scaler.inverse_transform(test_Y.reshape(-1, 1))

# Calculate mean squared error and mean absolute error on the test set
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
plt.ylabel('Add-to-cart Rate')
plt.legend()
plt.title('Add-to-cart Rate Predictions - ANN Model')

# Format the x-axis to show only the month
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

plt.gcf().autofmt_xdate()  # Auto-format the x-axis labels for better readability
plt.show()
