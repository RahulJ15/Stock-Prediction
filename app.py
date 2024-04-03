import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
from newsapi import NewsApiClient
import tensorflow as tf

st.title('Stock Trend Prediction')


# Function to fetch current news related to the stock
def get_stock_news(ticker, num_articles=5):
    try:
        newsapi = NewsApiClient(api_key='c68de8478ece4a258c4823f31671fe11')  # Replace 'YOUR_NEWS_API_KEY' with your actual API key
        top_headlines = newsapi.get_everything(q=ticker, language='en', sort_by='publishedAt')
        articles = top_headlines['articles'][:num_articles]
        return articles
    except:
        return None

# Date Selection
start_date = st.date_input("Select Start Date", pd.to_datetime('2019-01-01'))
end_date = st.date_input("Select End Date", pd.to_datetime('2024-01-01'))

start = start_date.strftime('%Y-%m-%d')
end = end_date.strftime('%Y-%m-%d')

user_input = st.text_input("Enter Stock Ticker", 'AAPL').upper()


# Display current news related to the stock in the sidebar
st.sidebar.title('Current News')
articles = get_stock_news(user_input, num_articles=5)
if articles:
    for article in articles:
        st.sidebar.write(f"- [{article['title']}]({article['url']})")
else:
    st.sidebar.write("Failed to retrieve current news. Please try again later.")

# Download data from Yahoo Finance
df = yf.download(user_input, start=start, end=end)
df = df.reset_index()

st.subheader(f'Data from {start} to {end}')
stats_to_display = df.describe().drop(['count', '25%', '50%', '75%'], axis=0)  # Removing count and quartiles
st.write(stats_to_display)
st.write("""
The data table provides a summary of statistical measures for the stock data within the selected date range. It includes the mean, standard deviation, minimum, and maximum values of the stock's closing prices. These statistics offer insights into the central tendency, variability, and range of the stock prices during the specified period, helping users understand the distribution and behavior of the stock's prices over time.
""")

# Closing Price vs Time Chart
st.subheader('Closing Price vs Time Chart')
fig1 = px.line(df, x='Date', y='Close', title='Closing Price vs Time')
st.plotly_chart(fig1)
st.write("This chart displays the closing prices of the selected stock over time. It helps visualize the general trend and fluctuations in the stock's price.")

# Closing Price vs Time Chart with 100 days MA
st.subheader('Closing Price vs Time Chart with 100 days MA')
ma_days = st.slider('Select Moving Average Days (1-365)', 1, 365, 100)
df[f'MA{ma_days}'] = df['Close'].rolling(window=ma_days).mean()
fig2 = px.line(df, x='Date', y=['Close', f'MA{ma_days}'], title=f'Closing Price vs Time with {ma_days} days MA')
st.plotly_chart(fig2)
st.write(f"This chart overlays the closing price with a {ma_days}-day moving average. Moving averages help smooth out price fluctuations and identify trends over time.")

# Model Training Data Shape
st.subheader('Model Training Data Shape')
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
st.write(data_training.shape)

# Model Testing Data Shape
st.subheader('Model Testing Data Shape')
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])
st.write(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Ensure data_training and data_testing have at least one column
if len(data_training.columns) == 0:
    data_training = data_training.assign(dummy=1)  # Add a dummy column if necessary
if len(data_testing.columns) == 0:
    data_testing = data_testing.assign(dummy=1)

data_training_array = scaler.fit_transform(data_training)

# Loading model
custom_objects = {'OrthogonalInitializer': tf.keras.initializers.Orthogonal}
model = tf.keras.models.load_model("LSTM_model.h5", custom_objects=custom_objects)


past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

# Option 2: Use scaler.fit_transform (recommended)
input_data = scaler.fit_transform(final_df)

x_test = np.array([input_data[i - 100:i] for i in range(100, input_data.shape[0])])
y_test = input_data[100:, 0]

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scale_factor = scaler.data_range_[0]  # Use the data range for scaling back

y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

prediction_df = pd.DataFrame({'Date': df['Date'].tail(len(y_test)), 'True Price': y_test.flatten(), 'Predicted Price': y_predicted.flatten()})

# Concatenate future predictions to prediction_df
future_dates = [(end_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]
future_predictions = []

for i in range(7):
    input_data = np.array(df['Close'].tail(100 + i))
    if len(input_data) < 100:  
        # Pad the input_data with the last observed value to make it 100 data points
        last_observed_value = df['Close'].iloc[-1]
        input_data = np.pad(input_data, (0, 100 - len(input_data)), 'edge')
        input_data[-(100 + i):] = last_observed_value
    else:
        # Slice input_data to include only the last 100 data points
        input_data = input_data[-100:]
    
    input_data = input_data.reshape(-1, 1)  # Reshape for single feature
    input_data = scaler.transform(input_data)  # Use the fitted scaler
    input_data = input_data.reshape(1, 100, 1)  # Reshape for LSTM model
    predicted_price = model.predict(input_data)
    predicted_price = predicted_price * scale_factor
    
    # Apply randomness to the predicted price
    min_percentage_change = 0.02  # -2%
    max_percentage_change = 0.05  # +5%
    random_percentage = np.random.uniform(min_percentage_change, max_percentage_change)
    sign = np.random.choice([-1, 1])  # Randomly choose between -1 and 1 for positive or negative change
    predicted_price *= (1 + sign * random_percentage)
    
    future_predictions.append(predicted_price[0][0])

future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})
prediction_df = pd.concat([prediction_df, future_df], ignore_index=True)

# Predicted v/s True
st.subheader('Predicted v/s True')
fig3 = px.line(prediction_df, x='Date', y=['True Price', 'Predicted Price'], title='Predicted v/s True Price')
st.plotly_chart(fig3)
st.write("This chart compares the true closing prices of the stock with the predicted prices generated by the LSTM model. It helps assess the accuracy of the model's predictions.")

# Display table for the next 7 days prediction
st.subheader('Next 7 Days Predicted Prices')
st.table(future_df)
st.write("This table displays the predicted prices for the next 7 days based on the LSTM model. Please note that these predictions include some randomness to account for market volatility.")
