from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import pandas as pd
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.dates as mdates
import numpy as np

# Load the data from the CSV file
df = pd.read_csv('add_to_cart_2_years.csv')

# Parse the 'date' column as datetime and set it as the index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Display the first few rows of the dataset
print(df.head())

# Split the data into train and test sets
split_index = int(len(df) * 0.8)
train = df['add_to_cart'].iloc[:split_index]
test = df['add_to_cart'].iloc[split_index:]

# Use auto_arima to select optimal parameters with seasonality
auto_arima = pm.auto_arima(train, seasonal=True, m=7, trace=True, stepwise=True, suppress_warnings=True, max_p=10, max_q=10)

print("Selected ARIMA parameters:", auto_arima.order)
print("Selected Seasonal parameters:", auto_arima.seasonal_order)

# Fit ARIMA model with selected parameters
model = ARIMA(train, order=auto_arima.order, seasonal_order=auto_arima.seasonal_order)
model_fit = model.fit()

# Rolling forecast
history = train.values.tolist()
predictions = []

for t in range(len(test)):
    model = ARIMA(history, order=auto_arima.order, seasonal_order=auto_arima.seasonal_order)
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    print('predicted=%f, expected=%f' % (yhat, test.values[t]))
    predictions.append(yhat)
    history.append(test.values[t])

# Remove first 10 observations to have same as ANN
predictions = predictions[10:]
test = test[10:]

# Evaluate forecasts using mean squared error
mse = mean_squared_error(test, predictions)
mae = mean_absolute_error(test, predictions)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((test - predictions) / test)) * 100

print('Test MAE: %.10f' % mae)
print('Test RMSE: %.10f' % rmse)
print('Test MAPE: %.10f' % mape)

# Plot forecasts against actual outcomes
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Add-to-cart Rate')
plt.title('Add-to-cart Rate Predictions - ARIMA Model')
plt.legend()

# Format the x-axis to show only the month
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

plt.gcf().autofmt_xdate()  # Auto-format the x-axis labels for better readability
plt.show()

print(f'avg test {sum(test) / len(test)}')
print(f'avg predictions {sum(predictions) / len(predictions)}')
