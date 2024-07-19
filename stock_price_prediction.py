import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import time
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# Streamlit app title
st.title('Stock Price Prediction using GRU')

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file, index_col='Date', parse_dates=['Date'])
    st.write(data.head())

    # Data visualization
    st.subheader('Stock Data Visualization')
    st.line_chart(data['Close'])

    # Data preprocessing
    price_data = data[['Close']]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    price_data.loc[:, 'Close'] = scaler.fit_transform(price_data['Close'].values.reshape(-1, 1))

    def split_data(stock, lookback):
        data_raw = stock.to_numpy() 
        data = []

        for index in range(len(data_raw) - lookback):
            data.append(data_raw[index: index + lookback])

        data = torch.tensor(data)
        test_set_size = int(0.2 * data.shape[0])
        train_set_size = data.shape[0] - test_set_size

        x_train = data[:train_set_size, :-1, :]
        y_train = data[:train_set_size, -1, :]

        x_test = data[train_set_size:, :-1]
        y_test = data[train_set_size:, -1, :]

        return x_train, y_train, x_test, y_test

    lookback = 20
    x_train, y_train, x_test, y_test = split_data(price_data, lookback)

    input_dim = 1
    hidden_dim = 32
    num_layers = 2
    output_dim = 1
    num_epochs = 105

    class GRU(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
            super(GRU, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
            out, _ = self.gru(x, h0.detach())
            out = self.fc(out[:, -1, :])
            return out

    model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    hist = torch.zeros(num_epochs)
    start_time = time.time()

    for t in range(num_epochs):
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train)
        hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    training_time = time.time() - start_time
    st.write(f"Training time: {training_time:.2f} seconds")

    # Predictions
    y_train_pred = model(x_train)
    y_test_pred = model(x_test)

    # Debug information
    st.write("Shape of y_train_pred:", y_train_pred.shape)
    st.write("Shape of y_train:", y_train.shape)

    # Ensure predictions are 2D
    y_train_pred = y_train_pred.view(-1, 1)
    y_test_pred = y_test_pred.view(-1, 1)

    # Convert back to original scale
    y_train_pred = torch.tensor(scaler.inverse_transform(y_train_pred.detach()))
    y_train = torch.tensor(scaler.inverse_transform(y_train.detach()))
    y_test_pred = torch.tensor(scaler.inverse_transform(y_test_pred.detach()))
    y_test = torch.tensor(scaler.inverse_transform(y_test.detach()))

    # Calculate RMSE
    trainScore = torch.sqrt(torch.mean((y_train[:, 0] - y_train_pred[:, 0])**2))
    testScore = torch.sqrt(torch.mean((y_test[:, 0] - y_test_pred[:, 0])**2))
    st.write(f'Train Score: {trainScore.item():.2f} RMSE')
    st.write(f'Test Score: {testScore.item():.2f} RMSE')

    # Prepare data for plotting
    trainPredictPlot = torch.full((price_data.shape[0], 1), float('nan'))
    trainPredictPlot[lookback:len(y_train_pred) + lookback, :] = y_train_pred

    testPredictPlot = torch.full((price_data.shape[0], 1), float('nan'))
    testPredictPlot[len(y_train_pred) + lookback - 1:len(price_data) - 1, :] = y_test_pred

    original = torch.tensor(scaler.inverse_transform(price_data['Close'].values.reshape(-1, 1)))

    # Combine predictions and original data
    predictions = torch.cat((trainPredictPlot, testPredictPlot, original), dim=1)
    result = pd.DataFrame(predictions.numpy(), columns=['Train Prediction', 'Test Prediction', 'Actual Value'], index=price_data.index)

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.index, y=result['Train Prediction'], mode='lines', name='Train Prediction'))
    fig.add_trace(go.Scatter(x=result.index, y=result['Test Prediction'], mode='lines', name='Test Prediction'))
    fig.add_trace(go.Scatter(x=result.index, y=result['Actual Value'], mode='lines', name='Actual Value'))
    fig.update_layout(
        xaxis=dict(showline=True, showgrid=True, showticklabels=True, linecolor='white', linewidth=2),
        yaxis=dict(title_text='Close (USD)', titlefont=dict(family='Rockwell', size=12, color='white'),
                   showline=True, showgrid=True, showticklabels=True, linecolor='white', linewidth=2, ticks='outside',
                   tickfont=dict(family='Rockwell', size=12, color='white')),
        showlegend=True, template='plotly_dark',
        title='Stock Price Prediction',
        font=dict(family='Rockwell', size=12, color='white')
    )
    st.plotly_chart(fig)