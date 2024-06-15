from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import pandas as pd
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.dates as mdates

# Load the data from the CSV file
df = pd.read_csv('conversion_rate_2_years.csv')

# Parse the 'date' column as datetime and set it as the index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Display the first few rows of the dataset
print(df.head())

# Split the data into train and test sets
split_index = int(len(df) * 0.8)
train = df['conversion_rate'].iloc[:split_index]
test = df['conversion_rate'].iloc[split_index:]

# Use auto_arima to select optimal parameters
auto_arima = pm.auto_arima(train, seasonal=True, trace=True, stepwise=True, suppress_warnings=True, max_p=10, max_q=10)

# Fit ARIMA model with selected parameters
model = ARIMA(train, order=auto_arima.order)
model_fit = model.fit()

# Rolling forecast
history = train.values.tolist()
predictions = []

for t in range(len(test)):
    model = ARIMA(history, order=auto_arima.order)
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    print('predicted=%f, expected=%f' % (yhat, test.values[t]))
    predictions.append(yhat)
    history.append(test.values[t])

# Remove first 8 observations to have same as ANN
predictions = predictions[8:]
test = test[8:]

# Evaluate forecasts using mean squared error
mse = mean_squared_error(test, predictions)
mae = mean_absolute_error(test, predictions)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((test - predictions) / test)) * 100

print('Test MSE: %.10f' % mse)
print('Test MAE: %.10f' % mae)
print('Test RMSE: %.10f' % rmse)
print('Test MAPE: %.10f' % mape)

# Plot forecasts against actual outcomes
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Conversion Rate')
plt.title('Conversion Rate Predictions - ARIMA Model')
plt.legend()

# Format the x-axis to show only the month
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

plt.gcf().autofmt_xdate()  # Auto-format the x-axis labels for better readability
plt.show()

print(f'avg test {sum(test) / len(test)}')
print(f'avg predictions {sum(predictions) / len(predictions)}')
