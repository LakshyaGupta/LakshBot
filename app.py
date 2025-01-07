import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import datetime
from io import BytesIO
from hashlib import sha256

st.set_page_config(page_title="Stock Prediction App", layout="wide")

PASSWORD = "securepassword"
def password_protect():
    """Simple password protection for the app."""
    password = st.text_input("Enter the password to access the app:", type="password")
    hashed_input = sha256(password.encode()).hexdigest()
    if hashed_input == sha256(PASSWORD.encode()).hexdigest():
        return True
    else:
        st.warning("Incorrect password. Please try again.")
        return False

if not password_protect():
    st.stop()

def main():
    option = st.sidebar.selectbox('Choose an option', ['Visualize', 'Predict', 'Analysis', 'Upload & Predict', 'Compare Predictions'])
    if option == 'Visualize':
        visualize_data()
    elif option == 'Predict':
        predict()
    elif option == 'Analysis':
        analyze_data()
    elif option == 'Upload & Predict':
        upload_and_predict()
    elif option == 'Compare Predictions':
        compare_predictions()

@st.cache_data
def download_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return df

def visualize_data():
    st.header('Visualize Stock Data')
    ticker = st.text_input('Enter Stock Ticker', 'AAPL').upper()
    start_date = st.date_input('Start Date', datetime.date(2010, 1, 1))
    end_date = st.date_input('End Date', datetime.date.today())
    if st.button('Load Data'):
        data = download_data(ticker, start_date, end_date)
        if not data.empty:
            st.line_chart(data['Close'])
            st.download_button("Download Data", data.to_csv().encode('utf-8'), file_name=f"{ticker}_data.csv", mime="text/csv")
        else:
            st.error('No data found for the selected ticker and date range.')

def predict():
    st.header('Predict Stock Prices')
    ticker = st.text_input('Enter Stock Ticker', 'AAPL').upper()
    start_date = st.date_input('Start Date', datetime.date(2010, 1, 1))
    end_date = st.date_input('End Date', datetime.date.today())
    prediction_days = st.number_input('Days to Predict', min_value=1, max_value=365, value=30)
    if st.button('Predict'):
        data = download_data(ticker, start_date, end_date)
        if not data.empty:
            st.write(f'Predicting the next {prediction_days} days for {ticker}')
            predictions, future_df = predict_stock(data, prediction_days)
            st.download_button("Download Predictions", future_df.to_csv().encode('utf-8'), file_name=f"{ticker}_predictions.csv", mime="text/csv")
        else:
            st.error('No data found for the selected ticker and date range.')

@st.cache_data
def predict_stock(data, prediction_days):
    data = data[['Close']].dropna()
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    training_data_len = int(len(dataset) * 0.8)

    train_data = scaled_data[:training_data_len]
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    test_data = scaled_data[training_data_len-60:]
    x_test, y_test = [], dataset[training_data_len:]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    future_predictions = []
    last_60_days = scaled_data[-60:]
    for _ in range(prediction_days):
        x_future = np.reshape(last_60_days, (1, 60, 1))
        pred_price = model.predict(x_future)[0, 0]
        future_predictions.append(pred_price)
        last_60_days = np.append(last_60_days[1:], [[pred_price]], axis=0)

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=prediction_days)
    future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Predicted Close'])

    return predictions, future_df

def analyze_data():
    st.header("Analyze Datasets")
    st.write("Under Construction: Advanced Analytics Coming Soon")

def upload_and_predict():
    st.header("Upload Datasets for Prediction")
    file = st.file_uploader("Upload a CSV file", type=["csv"])
    if file:
        data = pd.read_csv(file)
        st.write(data.head())

def compare_predictions():
    st.header("Compare Past Predictions to Actual Data")
    st.write("Under Construction: Comparison Coming Soon")

if __name__ == '__main__':
    main()
