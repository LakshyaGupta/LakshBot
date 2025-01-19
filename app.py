import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pymongo import MongoClient

st.set_page_config(page_title="Stock Analysis & Prediction Dashboard", layout="wide")

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["stock_predictions"]
collection = db["predictions"]

def main():
    st.title("Stock Analysis & Prediction Dashboard")

    st.sidebar.header("Dashboard Controls")
    dataset_url = st.sidebar.selectbox("Select Dataset URL", [
        "https://fred.stlouisfed.org/series/SP500",
        "https://fred.stlouisfed.org/series/DJIA",
        "https://fred.stlouisfed.org/series/NASDAQCOM",
        "https://fred.stlouisfed.org/series/NASDAQ100",
        "https://fred.stlouisfed.org/series/GDPC1",
        "https://fred.stlouisfed.org/series/FEDFUNDS",
        "https://fred.stlouisfed.org/series/UNRATE",
        "https://fred.stlouisfed.org/series/WM2NS",
        "https://fred.stlouisfed.org/series/DGS10",
        "https://fred.stlouisfed.org/series/DCOILWTICO"
    ])
    start_date = st.sidebar.date_input("Start Date", datetime(2010, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.today())
    prediction_years = st.sidebar.number_input("Years to Predict", min_value=1, max_value=10, value=1)
    timeframe = st.sidebar.selectbox("Select Timeframe", ["Daily", "Weekly", "Monthly", "Yearly"])

    if st.sidebar.button("Run Analysis"):
        data = download_data_from_fred(dataset_url, start_date, end_date)
        if not data.empty:
            st.header(f"Analysis for Dataset from {dataset_url}")
            visualize_data(data, timeframe)
            predictions, future_df = predict_stock(data, prediction_years)

            st.subheader("Prediction Results")
            st.write(future_df)
            st.download_button("Download Predictions as CSV", future_df.to_csv().encode("utf-8"), file_name="predictions.csv")

            save_to_mongodb(dataset_url, predictions, future_df)

            analyze_data(data, predictions, future_df, timeframe)
        else:
            st.error("No data found for the selected dataset and date range.")

def download_data_from_fred(url, start_date, end_date):
    series_id = url.split("/")[-1]
    api_key = "cdde234b4a095daa82255f352e612845"  # Replace with your actual FRED API key
    api_url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json&observation_start={start_date}&observation_end={end_date}"
    response = requests.get(api_url)
    if response.status_code == 200:
        observations = response.json().get("observations", [])
        data = pd.DataFrame(observations)
        data = data.rename(columns={"date": "Date", "value": "Close"})
        data["Date"] = pd.to_datetime(data["Date"])
        data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
        data = data.dropna().set_index("Date")
        return data
    else:
        st.error("Failed to fetch data from FRED API. Check your API key or series ID.")
        return pd.DataFrame()

def visualize_data(data, timeframe):
    data_resampled = resample_data(data, timeframe)
    st.subheader("Historical Data Visualization")
    st.line_chart(data_resampled['Close'])

def resample_data(data, timeframe):
    if timeframe == "Weekly":
        return data.resample('W').mean(numeric_only=True)
    elif timeframe == "Monthly":
        return data.resample('M').mean(numeric_only=True)
    elif timeframe == "Yearly":
        return data.resample('Y').mean(numeric_only=True)
    else:
        return data

def predict_stock(data, prediction_years):
    prediction_days = prediction_years * 365  # Convert years to days
    data = data[['Close']].dropna()
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data_len = int(len(dataset) * 0.8)
    train_data = scaled_data[:train_data_len]
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

    test_data = scaled_data[train_data_len-60:]
    x_test, y_test = [], dataset[train_data_len:]
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

def save_to_mongodb(dataset_url, predictions, future_df):
    data_to_save = {
        "dataset_url": dataset_url,
        "predictions": predictions.tolist(),
        "future_predictions": future_df.to_dict()
    }
    collection.insert_one(data_to_save)
    st.success("Predictions saved to MongoDB.")

def analyze_data(data, predictions, future_df, timeframe):
    st.subheader("Analysis")
    data_resampled = resample_data(data, timeframe)
    std_devs = np.std(data_resampled['Close'])
    mean_val = np.mean(data_resampled['Close'])
    st.write(f"Historical Mean: {mean_val}, Standard Deviation: {std_devs}")

    correlation_matrix = data.corr()
    st.write("### Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, ax=ax, cmap='coolwarm')
    st.pyplot(fig)

    st.write("### Patterns and Insights")
    st.write("Identify relationships, time lags, and deviations from historical means.")

if __name__ == '__main__':
    main()
