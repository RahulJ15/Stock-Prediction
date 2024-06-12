import streamlit as st
import hashlib
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
from newsapi import NewsApiClient
import tensorflow as tf
import time

# Hashing function for passwords
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

# Check the hashed password
def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

# Create or connect to an SQLite database
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create users table
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    username TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL
)
''')

# Create user_stocks table
c.execute('''
CREATE TABLE IF NOT EXISTS user_stocks (
    username TEXT NOT NULL,
    stock TEXT NOT NULL,
    PRIMARY KEY (username, stock),
    FOREIGN KEY (username) REFERENCES users (username)
)
''')
conn.commit()

# Fetch user details from the database
def fetch_user_details(username):
    c.execute('SELECT username, password FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    return user

# Add new user to the database
def add_user(username, password):
    c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, make_hashes(password)))
    conn.commit()

# Add stock to user's list
def add_user_stock(username, stock):
    # Check if the stock already exists in the user's list
    if stock in fetch_user_stocks(username):
        st.error(f"{stock} is already in your favourites")
        return
    
    # If the stock doesn't exist, insert it into the database
    c.execute('INSERT INTO user_stocks (username, stock) VALUES (?, ?)', (username, stock))
    conn.commit()


# Fetch user's stock list
def fetch_user_stocks(username):
    c.execute('SELECT stock FROM user_stocks WHERE username = ?', (username,))
    stocks = c.fetchall()
    return [stock[0] for stock in stocks]

# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = "AAPL"

def login(username):
    st.session_state.logged_in = True
    st.session_state.username = username

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""


def remove_user_stock(username, stock):
    c.execute('DELETE FROM user_stocks WHERE username = ? AND stock = ?', (username, stock))
    conn.commit()


def get_current_price(ticker):
    stock = yf.Ticker(ticker)
    current_price = stock.history(period='1d')['Close'].iloc[-1]
    return current_price

def app_content(username):
    st.title(f"Welcome {username}")


    search_input = st.text_input("Search for a Stock Ticker", value=st.session_state.selected_stock.upper())

    # Update the selected stock based on user input
    if search_input.strip().upper() != st.session_state.selected_stock.upper():
        st.session_state.selected_stock = search_input.strip().upper()
        st.experimental_rerun()

    st.sidebar.title('Your Stocks')
    user_stocks = fetch_user_stocks(username)
    for stock in user_stocks:
        cols = st.sidebar.columns([3, 2])
        if cols[0].button(stock, key=f"select_{stock}"):
            st.session_state.selected_stock = stock
            
        if cols[1].button("Remove", key=f"remove_{stock}"):
            remove_user_stock(username, stock)
            st.sidebar.success(f"Removed {stock} from your favourites")
            

    new_stock = st.sidebar.text_input("Add a new stock to your favourites")
    if st.sidebar.button("Add Stock"):
        add_user_stock(username, new_stock)
        st.sidebar.success(f"Added {new_stock} to your favourites")

    user_input = st.session_state.selected_stock

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
    end_date = st.date_input("Select End Date", datetime.now().date())

    start = start_date.strftime('%Y-%m-%d')
    end = end_date.strftime('%Y-%m-%d')

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


    price_placeholder = st.empty()

    def update_price():
        current_price = get_current_price(user_input)
        if current_price:
            if user_input.endswith('.NS') or user_input.endswith('.BO'):
                formatted_price = "₹{:.2f}".format(current_price)
            else:
                formatted_price = "${:.2f}".format(current_price)
            price_placeholder.metric("Current Price", formatted_price)
        else:
            st.error("Failed to fetch current price. Please try again later.")

    update_price()
    


    # Closing Price vs Time Chart with 100 days MA
    st.subheader('Closing Price vs Time Chart with Moving Average')
    ma_days = st.slider('Select Moving Average Days (1-365)', 1, 365, 100)
    df[f'MA{ma_days}'] = df['Close'].rolling(window=ma_days).mean()
    fig2 = px.line(df, x='Date', y=['Close', f'MA{ma_days}'], title=f'Closing Price vs Time with {ma_days} days MA')
    st.plotly_chart(fig2)
    st.write(f"This chart overlays the closing price with a {ma_days}-day moving average. Moving averages help smooth out price fluctuations and identify trends over time.")

    # Model Training Data Shape   
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])    
    # Model Testing 
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Ensure data_training and data_testing have at least one column
    if len(data_training.columns) == 0:
        data_training = data_training.assign(dummy=1)  # Add a dummy column if necessary
    if len(data_testing.columns) == 0:
        data_testing = data_testing.assign(dummy=1)

    data_training_array = scaler.fit_transform(data_training)

    # Loading model
    model = tf.keras.models.load_model("LSTM_model.h5")

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

    # Append future predictions to prediction_df, ensuring dates align and using only scaled predictions
    future_dates = [(end_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]
    future_predictions_scaled_down = []

    for i in range(7):
        input_data = np.array(df['Close'].tail(100 + i))
        if len(input_data) < 100:  
            last_observed_value = df['Close'].iloc[-1]
            input_data = np.pad(input_data, (0, 100 - len(input_data)), 'edge')
        else:
            input_data = input_data[-100:]
        
        input_data = input_data.reshape(-1, 1)
        input_data_scaled = scaler.transform(input_data)
        input_data_scaled = input_data_scaled.reshape(1, 100, 1)
        predicted_price_scaled = model.predict(input_data_scaled)
        
        # Scale down the predicted price
        predicted_price = scaler.inverse_transform(predicted_price_scaled).flatten()[0]

        # Apply randomness to the scaled predicted price
        random_change = np.random.uniform(-0.02, 0.02)
        predicted_price *= (1 - random_change)  # Change here to scale down
        
        future_predictions_scaled_down.append(predicted_price)

    # Prepare the DataFrame for plotting
    future_df_scaled_down = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions_scaled_down})

    # Append future predictions to prediction_df, ensuring dates align and using only scaled predictions
    prediction_df = pd.concat([prediction_df, future_df_scaled_down], ignore_index=True)

    # Plotting Predicted vs True Prices
    st.subheader('Predicted v/s True Price (Including Next 7 Days Predictions)')
    fig3 = px.line(prediction_df, x='Date', y=['True Price', 'Predicted Price'], title='Predicted v/s True Price')
    st.plotly_chart(fig3)

    # Next 7 Days Predicted Prices table display
    st.subheader('Next 7 Days Predicted Prices')
    st.table(future_df_scaled_down)

# Main app logic
def main():
    if st.session_state.logged_in:
        app_content(st.session_state.username)
        if st.sidebar.button("Logout"):
            logout()
            st.experimental_rerun()
    else:
        st.sidebar.title("Login/Signup")

        menu = ["Login", "Sign Up"]
        choice = st.sidebar.selectbox("Select an option", menu)

        if choice == "Login":
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type='password')

            if st.button("Login"):
                user = fetch_user_details(username)
                if user and check_hashes(password, user[1]):
                    st.success(f"Logged in as {username}")
                    login(username)
                    st.experimental_rerun()
                else:
                    st.warning("Incorrect Username/Password")

        elif choice == "Sign Up":
            st.subheader("Sign Up")
            new_user = st.text_input("New Username")
            new_password = st.text_input("New Password", type='password')

            if st.button("Sign Up"):
                user = fetch_user_details(new_user)
                if user:
                    st.warning("Username already exists")
                else:
                    add_user(new_user, new_password)
                    st.success("You have successfully created an account")
                    st.info("Please login to continue")

if __name__ == '__main__':
    main()
