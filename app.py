import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import datetime

st.title('Stock Price Prediction App')
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')
st.sidebar.info("Created and designed by [Lakshya Gupta](https://www.linkedin.com/in/lakshya-gupta-2004/)")

def main():
    option = st.sidebar.selectbox('Choose an option', ['Visualize', 'Predict'])
    if option == 'Visualize':
        visualize_data()
    elif option == 'Predict':
        predict()

@st.cache
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
            predict_stock(data, prediction_days)
        else:
            st.error('No data found for the selected ticker and date range.')

def predict_stock(data, prediction_days):
    # Prepare data
    data = data[['Close']]
    data = data.dropna()
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * 0.8))

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Create training data
    train_data = scaled_data[0:int(training_data_len), :]
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Create testing data
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get model predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    st.pyplot(plt)

    # Predict future prices
    last_60_days = scaled_data[-60:]
    future_predictions = []
    for _ in range(prediction_days):
        X_test = np.array([last_60_days])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        pred_price = model.predict(X_test)
        future_predictions.append(pred_price[0, 0])
        last_60_days = np.append(last_60_days[1:], pred_price, axis=0)
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Plot future predictions
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=prediction_days)
    future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Predicted Close'])

    plt.figure(figsize=(16, 8))
    plt.title('Future Predictions')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.plot(data['Close'])
    plt.plot(future_df['Predicted Close'])
    plt.legend(['Historical', 'Predicted'], loc='lower right')
    st.pyplot(plt)

if __name__ == '__main__':
    main()
