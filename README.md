
# Stock Trend Prediction with LSTM

This project utilizes Long Short-Term Memory (LSTM) networks to predict the future trend of stock prices. It employs historical stock price data obtained from Yahoo Finance and TensorFlow for building the LSTM model. The user interface is developed using Streamlit, allowing users to select a specific stock, choose a date range, visualize historical data, and view future price predictions.

## Getting Started

To get started with this project, ensure you have the necessary dependencies installed. You can install them using pip:

```bash
pip install pandas numpy plotly yfinance tensorflow streamlit
```

Additionally, ensure you have a trained LSTM model (saved as `LSTM_model.h5`) in the project directory.

## Running the Application

You can run the application locally by executing the following command:

```bash
streamlit run app.py
```

This command will start a local server, and you can access the application in your web browser by visiting `http://localhost:8501`.

## Usage

1. **Date Selection**: Select the start and end dates for the historical stock price data.
2. **Enter Stock Ticker**: Input the ticker symbol of the desired stock (e.g., AAPL for Apple Inc.).
3. **Visualize Data**: View descriptive statistics of the selected stock's closing prices and visualize them using interactive charts.
4. **Model Prediction**: See the predicted prices for the next 7 days based on the LSTM model.
5. **Stay Informed**: Get access to the latest news related to the selected stock, providing valuable insights for informed decision-making.

## Features

- **Data Visualization**: Visualize historical stock price data using interactive charts.
- **Predictive Analysis**: Utilize LSTM model to predict future stock prices.
- **User-friendly Interface**: Streamlit-based UI for easy interaction and visualization.
- **Descriptive Statistics**: View descriptive statistics of stock prices within the selected date range.
- **Dynamic Moving Average**: Overlay closing price chart with a dynamic moving average.
- **Future Price Prediction**: Display predicted prices for the next 7 days.
- **Current News Integration**: Fetch and display current news related to the selected stock using the NewsAPI, offering users up-to-date information for better decision-making.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

