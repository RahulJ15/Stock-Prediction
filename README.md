Certainly! Here's an updated README file that combines both the description of the application features and the usage of the database:

---

# Stock Trend Prediction with LSTM

This project utilizes Long Short-Term Memory (LSTM) networks to predict the future trend of stock prices. It employs historical stock price data obtained from Yahoo Finance and TensorFlow for building the LSTM model. The user interface is developed using Streamlit, allowing users to select a specific stock, choose a date range, visualize historical data, and view future price predictions. User authentication and personalized stock lists are managed using SQLite as the database management system.

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

## Features

1. **User Registration**:
   - New users can sign up for an account by providing a unique username and password.
   - User credentials are securely stored in an SQLite database, ensuring data integrity and security.

2. **User Authentication**:
   - Registered users can log in using their username and password.
   - Authentication is performed against the stored credentials in the SQLite database.

3. **Date Selection**: 
   - Select the start and end dates for the historical stock price data.

4. **Enter Stock Ticker**:
   - Input the ticker symbol of the desired stock (e.g., AAPL for Apple Inc.).

5. **Visualize Data**:
   - View descriptive statistics of the selected stock's closing prices and visualize them using interactive charts.

6. **Model Prediction**:
   - See the predicted prices for the next 7 days based on the LSTM model.

7. **Personalized Stock Lists**:
   - Each user can maintain a personalized list of stocks.
   - Add new stocks to the list or remove existing ones, with changes reflected in the SQLite database.

8. **Stay Informed**:
   - Get access to the latest news related to the selected stock, providing valuable insights for informed decision-making.



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

This README provides comprehensive information about the project, including its features, usage, and integration with the SQLite database for user authentication and personalized stock lists.