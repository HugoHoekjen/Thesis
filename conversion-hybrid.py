import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from keras.optimizers import Adam
import pmdarima as pm
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import random
import matplotlib.dates as mdates
from sklearn.model_selection import TimeSeriesSplit



# Set seeds for reproducibility
np.random.seed(333)
random.seed(333)
tf.random.set_seed(333)

# Load data
df = pd.read_csv('conversion_rate_2_years.csv')

# Parse the 'date' column as datetime and set it as the index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Extract the 'visits' column
series = df['conversion_rate']

# Split in test and training sets
split_index = int(len(series) * 0.8)
train = series.iloc[:split_index].tolist()
test = series.iloc[split_index:].tolist()

auto_arima = pm.auto_arima(train, seasonal=True, stepwise=True, suppress_warnings=True, max_p=10, max_q=10)
print("Selected ARIMA parameters:", auto_arima.order)

# Fit ARIMA model on training set and extract the residuals
arima_model_train = ARIMA(train, order=auto_arima.order)
model_fit_arima_train = arima_model_train.fit()
train_residuals = pd.DataFrame(model_fit_arima_train.resid)

# ARIMA Rolling Forecast on the test set
arima_test_predictions = []
history = train.copy()
for t in range(len(test)):
    model = ARIMA(history, order=auto_arima.order)
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    arima_test_predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

# Calculate residuals
test_resid = [test[i] - arima_test_predictions[i] for i in range(len(test))]

# Evaluate forecasts using mean squared error
mae_test_arima = mean_absolute_error(test, arima_test_predictions)

print('ARIMA Test MAE: %.10f' % mae_test_arima)

#####################################
########## HYBRID ###################
#####################################

# Create ANN model
window_size = 15
def make_model(window_size):
    model = Sequential()
    model.add(Dense(10, input_dim=window_size, kernel_initializer="uniform", activation="sigmoid"))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=Adam())
    return model

model = make_model(window_size)

# Scale data
min_max_scaler = MinMaxScaler()
train_scaled = min_max_scaler.fit_transform(train_residuals)

# Create train X and train Y
train_X, train_Y = [], []
for i in range(len(train_scaled) - window_size):
    train_X.append(train_scaled[i:i + window_size])
    train_Y.append(train_scaled[i + window_size])

# Reshape training data and create numpy arrays
new_train_X = np.array([x.reshape(-1) for x in train_X])
new_train_Y = np.array([y.reshape(-1) for y in train_Y])

# # Implement time series cross-validation on the training set -- commented out for speed as unused after hyperparameter tuning
# tscv = TimeSeriesSplit(n_splits=5)
#
# mse_scores = []
# mae_scores = []
#
# for train_index, val_index in tscv.split(new_train_X):
#     split_train_X, val_X = new_train_X[train_index], new_train_X[val_index]
#     split_train_Y, val_Y = new_train_Y[train_index], new_train_Y[val_index]
#
#     # Build and train the model
#     model = make_model(window_size)
#     history = model.fit(split_train_X, split_train_Y, epochs=100, batch_size=64, validation_data=(val_X, val_Y), verbose=0)
#
#     # Make predictions
#     predictions = model.predict(val_X)
#
#     # Inverse scaling
#     predictions_rescaled = min_max_scaler.inverse_transform(predictions)
#     val_Y_rescaled = min_max_scaler.inverse_transform(val_Y.reshape(-1, 1))
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

# Fit the model on the training set
model.fit(new_train_X, new_train_Y, epochs=750, batch_size=128, validation_split=0.05)

# Extend the test set to include the last window_size observations of the training set for rolling window approach
test_extended = train_residuals.values[-window_size:].flatten().tolist() + test_resid

# Scale data
test_scaled = min_max_scaler.fit_transform(np.array(test_extended).reshape(-1, 1))

# Create test X
test_X = []
for i in range(len(test_scaled) - window_size):
    test_X.append(test_scaled[i:i + window_size])

new_test_X = np.array([x.reshape(-1) for x in test_X])

# Predict model on the test set and scale data back to its original format
predictions = model.predict(new_test_X)
ann_predictions_rescaled = min_max_scaler.inverse_transform(predictions).squeeze().tolist()

# Sum ARIMA predictions and ANN residual predictions
pred_final = [ann + arima for ann, arima in zip(ann_predictions_rescaled, arima_test_predictions)]

# Remove first 8 observations to have same as ANN
test = test[8:]
pred_final = pred_final[8:]

# Calculate total mae
total_mae = mean_absolute_error(test, pred_final)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test, pred_final))

# Calculate MAPE
mape = mean_absolute_percentage_error(test, pred_final)

print('Test MAE total prediction: %.10f' % total_mae)
print('Test RMSE total prediction: %.10f' % rmse)
print('Test MAPE total prediction: %.10f' % mape)

# Plot actual vs predicted values for ARIMA
test_dates = df.index[split_index + 8:]
test_resid = test_resid[8:]
ann_predictions_rescaled = ann_predictions_rescaled[8:]

# Plot ANN predicted residuals against the actual ARIMA residuals
plt.plot(test_dates, test_resid, label='ARIMA Residuals', color='blue')
plt.plot(test_dates, ann_predictions_rescaled, label='ANN Predicted Residuals', color='red')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.legend()
plt.title('ANN Predicted Residuals vs. ARIMA Residuals')
# Format the x-axis to show only the month
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

plt.gcf().autofmt_xdate()  # Auto-format the x-axis labels for better readability
plt.show()

# Plot final predictions against actual test set
plt.plot(test_dates, test, label='Actual')
plt.plot(test_dates, pred_final, label='Final Predicted', color='red')
plt.xlabel('Date')
plt.ylabel('Conversion Rate')
plt.legend()
plt.title('Conversion Rate predictions - ARIMA-ANN model')
# Format the x-axis to show only the month
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

plt.gcf().autofmt_xdate()  # Auto-format the x-axis labels for better readability
plt.show()
