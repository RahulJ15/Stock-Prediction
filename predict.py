import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import tensorflow as tf
from keras.utils import pad_sequences
import matplotlib.dates as mdates

# Function to download stock data
def download_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    return df.reset_index()

# Function to plot Closing Price vs Time Chart
def plot_closing_price(df, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Date'], df['Close'])
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    st.pyplot(fig)

# Function to plot Closing Price vs Time Chart with Moving Average
def plot_ma_chart(df, window_size, title):
    ma = df['Close'].rolling(window_size).mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Date'], ma, label=f'{window_size}-day MA')
    ax.plot(df['Date'], df['Close'], label='Closing Price')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_title(title)
    st.pyplot(fig)

# Function to plot Predicted v/s True
def plot_predicted_vs_true(df, y_predicted, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Date'][100:], df['Close'][100:], 'b', label='Original Price')
    ax.plot(df['Date'][100:len(y_predicted)+100], y_predicted, 'r', label='Predicted Price')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_title(title)
    ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    st.pyplot(fig)

# Main Streamlit app
def main():
    st.title('Stock Trend Prediction')

    # User input for stock ticker
    user_input = st.text_input("Enter Stock Ticker", 'AAPL')

    # User-defined time range
    start_date = st.date_input("Select start date", pd.to_datetime("2023-01-01"))
    end_date = st.date_input("Select end date", pd.to_datetime("2024-01-01"))

    # Download stock data based on user-defined time range
    df = download_stock_data(user_input, start_date, end_date)

    st.subheader(f'Data from {start_date} to {end_date}')
    st.write(df.describe())

    # Select specific date
    selected_date = st.date_input("Select a date for detailed information", pd.to_datetime("2024-01-01"))

    st.markdown("""
    ### Moving Average

    A Moving Average is a statistical tool used to smooth out fluctuations in data over time. In stock market analysis, it's often employed to identify trends by averaging out short-term price movements.

    The user can adjust the 'Select moving average window size' slider to customize the period over which the Moving Average is calculated. This chart displays the Closing Price along with the Moving Average, helping users visualize trends and potential turning points.

    """)

    # Closing Price vs Time Chart
    plot_closing_price(df, 'Closing Price vs Time Chart')

    # Customizable Moving Average
    ma_window = st.slider("Select moving average window size", min_value=1, max_value=365, value=100)
    plot_ma_chart(df, ma_window, f'Closing Price vs Time Chart with {ma_window}-day MA')

    # Data splitting for training and testing
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

    # Data preprocessing
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    # Loading pre-trained model
    model = tf.keras.models.load_model("LSTM_model.h5")

    # User-defined prediction horizon
    horizon_days = st.slider("Select prediction horizon (days)", min_value=1, max_value=30, value=7)

    # Prepare input data for prediction with customizable horizon
    past_100_days = data_training.tail(100)
    future_days = data_testing.head(horizon_days)

    # Combine 'Date' columns from past_100_days and future_days
    input_data = pd.concat([past_100_days, future_days[['Date', 'Close']]], axis=0, ignore_index=True)

    # Convert 'Date' column to datetime
    input_data['Date'] = pd.to_datetime(input_data['Date'])

    # Convert 'Close' column to numeric (in case it's not already)
    input_data['Close'] = pd.to_numeric(input_data['Close'])

    # Scale the 'Close' column
    input_data_array = scaler.transform(input_data[['Close']])
    x_test = [input_data_array[i - 100:i, 0] for i in range(100, input_data_array.shape[0])]
    x_test = np.array(x_test)
    y_test = input_data_array[100:, 0]

    # Predict using the model
    y_predicted = model.predict(x_test)

    # Inverse transform to get the actual predicted values
    scaler_custom = scaler.scale_
    scale_factor_custom = 1 / scaler_custom[0]
    y_predicted_custom = y_predicted * scale_factor_custom

    # Append 'Date' column to final_df before assigning 'Predicted' column
    final_df = pd.concat([past_100_days, data_testing.head(horizon_days)], ignore_index=True)
    final_df['Date'] = pd.to_datetime(final_df['Date'])  # Convert 'Date' column to datetime
    final_df['Predicted'] = np.nan
    final_df['Predicted'][100:100 + horizon_days] = y_predicted_custom.flatten()

    # Plot Predicted v/s True with customizable horizon
    y_test_custom = y_test * scale_factor_custom
    plot_predicted_vs_true(final_df, y_predicted_custom.flatten(), f'Predicted v/s True for {horizon_days}-day Horizon')

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
