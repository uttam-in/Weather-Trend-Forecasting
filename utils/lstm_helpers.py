import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from utils.plot_time_series import filter_data_country_location


# PyTorch LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Function to train and forecast using LSTM
def forecast_with_lstm(data, field, steps, epochs):
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[field].dropna().values.reshape(-1, 1))

    # Prepare the data for LSTM
    sequence_length = 10  # Number of time steps
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = torch.FloatTensor(X).unsqueeze(2)  # Add a feature dimension
    y = torch.FloatTensor(y)

    # Split into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Define the LSTM model
    input_dim = 1
    hidden_dim = 50
    num_layers = 2
    output_dim = 1
    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
    model = model.to(torch.device('cpu'))  # Change to 'cuda' if GPU is available

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(epochs):  # Use the input epochs here
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train.unsqueeze(1))
        loss.backward()
        optimizer.step()

    # Make predictions
    model.eval()
    predictions = model(X_test).detach().numpy()
    predictions = scaler.inverse_transform(predictions)  # Scale back to original values
    y_test = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))  # Scale back test values

    mse = mean_squared_error(y_test, predictions)

    # Prepare data for plotting
    train_data = scaler.inverse_transform(scaled_data[:train_size].reshape(-1, 1))
    test_data = y_test
    forecast_index = data.index[train_size + sequence_length:]

    return train_data, test_data, predictions, forecast_index, mse

# Function to update the plot with LSTM forecast
def update_plot_with_lstm(data, output, country, location_name, field, steps, epochs):
    data_filtered = filter_data_country_location(data, country, location_name)

    try:
        train_data, test_data, predictions, forecast_index, mse = forecast_with_lstm(data_filtered, field, steps, epochs)
    except Exception as e:
        with output:
            output.clear_output()
            print(f"Error: {e}")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(data_filtered.index[:len(train_data)], train_data, label='Training Data', color='blue')
    plt.plot(forecast_index, test_data, label='Actual Test Data', color='green')
    plt.plot(forecast_index, predictions, label='Forecast', color='red', linestyle='dashed')
    plt.title(f'{field.replace("_", " ").title()} Forecast (LSTM) - {country}, {location_name}')
    plt.xlabel('Date')
    plt.ylabel(field.replace("_", " ").title())
    plt.legend()
    plt.show()

    print(f"Mean Squared Error: {mse:.4f}")

def lstm_forecast(data, field, lags, steps):
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[field].dropna().values.reshape(-1, 1))

    # Prepare the data for LSTM
    X, y = [], []
    for i in range(lags, len(scaled_data)):
        X.append(scaled_data[i - lags:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = torch.FloatTensor(X).unsqueeze(2)
    y = torch.FloatTensor(y)

    # Train-Test Split
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]

    # Define LSTM Model
    model = LSTMModel(input_dim=1, hidden_dim=30, num_layers=2, output_dim=1)

    # Define Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the Model
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train.unsqueeze(1))
        loss.backward()
        optimizer.step()

    # Forecast
    forecast_values = list(y_train[-lags:].numpy())
    forecast_results = []
    for _ in range(steps):
        input_tensor = torch.FloatTensor(forecast_values[-lags:]).unsqueeze(0).unsqueeze(2)
        prediction = model(input_tensor).detach().numpy()[0][0]
        forecast_results.append(prediction)
        forecast_values.append(prediction)

    # Inverse transform to original scale
    return scaler.inverse_transform(np.array(forecast_results).reshape(-1, 1)).flatten()