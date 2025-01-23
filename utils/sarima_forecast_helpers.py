# SARIMA model
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Function to fit SARIMA and forecast
def forecast_with_sarima(data, field, p, d, q, P, D, Q, s, steps):
    values = data[field].dropna()
    train_size = int(len(values) * 0.8)
    train_data = values[:train_size]
    test_data = values[train_size:]

    # Fit SARIMA model
    model = SARIMAX(
        train_data,
        order=(p, d, q),
        seasonal_order=(P, D, Q, s),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_fitted = model.fit(disp=False)

    # Forecast
    forecast = model_fitted.forecast(steps=steps)
    forecast_index = test_data.index[:steps] if len(test_data) >= steps else test_data.index

    mse = mean_squared_error(test_data[:steps], forecast) if len(test_data) >= steps else None
    return train_data, test_data, forecast, forecast_index, mse

# Function to update the plot with SARIMA forecast
def update_plot_with_forecast_sarima(data, output, country, location_name, field, p, d, q, P, D, Q, s, steps):
    data_filtered = data[(data['country'] == country) & (data['location_name'] == location_name)]

    try:
        train_data, test_data, forecast, forecast_index, mse = forecast_with_sarima(
            data_filtered, field, p, d, q, P, D, Q, s, steps
        )
    except Exception as e:
        with output:
            output.clear_output()
            print(f"Error: {e}")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data, label='Training Data', color='blue')
    if len(test_data) > 0:
        plt.plot(test_data.index, test_data, label='Actual Test Data', color='green')
    plt.plot(forecast_index, forecast, label='Forecast', color='red', linestyle='dashed')
    plt.title(f'{field.replace("_", " ").title()} Forecast (SARIMA) - {country}, {location_name}')
    plt.xlabel('Date')
    plt.ylabel(field.replace("_", " ").title())
    plt.legend()
    plt.show()

    if mse is not None:
        print(f"Mean Squared Error: {mse:.4f}")
    else:
        print("Not enough test data to compute MSE.")

# SARIMA model
def sarima_forecast(train_data, steps, p, d, q, P, D, Q, s):
    model = SARIMAX(
        train_data,
        order=(p, d, q),
        seasonal_order=(P, D, Q, s),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_fitted = model.fit(disp=False)
    return model_fitted.forecast(steps=steps)