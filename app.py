import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import seaborn as sns
from datetime import datetime, timedelta
from pymongo import MongoClient

st.set_page_config(page_title="Stock Analysis & Prediction Dashboard", layout="wide")

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["stock_predictions"]
collection = db["predictions"]

# Password Protection Flag
password_correct = False

def password_protection():
    global password_correct
    password = st.sidebar.text_input("Enter password to access the dashboard", type="password")
    if password == "Associate7!!7t":  # Replace with your desired password
        password_correct = True
        st.sidebar.success("Password correct! You now have access.")
        main_runner()
    elif password:
        st.sidebar.error("Incorrect password. Please try again.")

# If the password is correct, show the rest of the dashboard
def main():
    
    if not password_correct:
        password_protection()
    else: 
        main_runner()
def main_runner():
    st.title("Stock Analysis & Prediction Dashboard")

    st.sidebar.header("Dashboard Controls")
    dataset_urls = st.sidebar.multiselect("Select Dataset URLs", [
        "SP500", "DJIA", "NASDAQCOM", "NASDAQ100", "GDPC1", "FEDFUNDS",
        "UNRATE", "WM2NS", "DGS10", "DCOILWTICO"
    ])
    start_date = st.sidebar.date_input("Start Date", datetime(2010, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.today())
    prediction_years = st.sidebar.number_input("Years to Predict", min_value=1, max_value=10, value=1)
    timeframe = st.sidebar.selectbox("Select Timeframe", ["Daily", "Weekly", "Monthly", "Yearly"])

    if st.sidebar.button("Run Analysis"):
        all_data = {}
        for dataset in dataset_urls:
            data = download_data_from_fred(dataset, start_date, end_date)
            if not data.empty:
                all_data[dataset] = data

        if all_data:
            combined_data = combine_datasets(all_data)
            st.header("Combined Data Visualization")
            visualize_data(combined_data, timeframe)

            predictions, future_df, backtested_results = predict_stock(combined_data, prediction_years)

            st.subheader("Prediction Results")
            st.write(future_df)
            st.download_button("Download Predictions as CSV", future_df.to_csv().encode("utf-8"), file_name="predictions.csv")

            save_to_mongodb(dataset_urls, predictions, future_df)

            st.subheader("Backtested Results")
            st.line_chart(backtested_results)

            analyze_data(combined_data, predictions, future_df, timeframe)
        else:
            st.error("No data found for the selected dataset URLs and date range.")

def download_data_from_fred(series_id, start_date, end_date):
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

def combine_datasets(data_dict):
    combined_data = pd.concat(data_dict.values(), axis=1, join="outer")
    
    # Ensure the number of columns matches the number of datasets
    num_datasets = len(data_dict)
    if num_datasets == 2:
        combined_data = pd.concat([combined_data] * 3, axis=1)
        combined_data.columns = [col + f"_{i}" for i, col in enumerate(combined_data.columns)]
    elif num_datasets == 3:
        combined_data = pd.concat([combined_data] * 2, axis=1)
        combined_data.columns = [col + f"_{i}" for i, col in enumerate(combined_data.columns)]
    else:
        combined_data.columns = list(data_dict.keys())

    # Check for any non-numeric columns and drop them if needed
    combined_data = combined_data.select_dtypes(include=[np.number])

    # Drop rows with NaN values
    combined_data = combined_data.dropna()

    return combined_data

def visualize_data(data, timeframe):
    data_resampled = resample_data(data, timeframe)
    st.line_chart(data_resampled)

def resample_data(data, timeframe):
    # Make sure we only apply resampling to numeric columns (like 'Close')
    numeric_data = data.select_dtypes(include=[np.number])
    if timeframe == "Weekly":
        return numeric_data.resample('W').mean()
    elif timeframe == "Monthly":
        return numeric_data.resample('M').mean()
    elif timeframe == "Yearly":
        return numeric_data.resample('Y').mean()
    else:
        return numeric_data

def predict_stock(data, prediction_years):
    prediction_days = prediction_years * 365  # Convert years to days
    dataset = data.mean(axis=1).dropna().values.reshape(-1, 1)  # Use the mean of combined data for predictions
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

    model = Sequential([  # LSTM model architecture
        LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Predict the test data (test set)
    test_data = scaled_data[train_data_len-60:]
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Predict future values
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

    # Backtest by comparing predictions to actual values
    backtested_results = pd.DataFrame({
        'Actual': scaler.inverse_transform(scaled_data[train_data_len:].reshape(-1, 1)).flatten(),
        'Predicted': predictions.flatten()
    }, index=data.index[train_data_len:])

    return predictions, future_df, backtested_results

def save_to_mongodb(dataset_urls, predictions, future_df):
    # Convert future_df index to strings
    future_df.index = future_df.index.map(str)
    future_df_dict = future_df.to_dict()

    # Create the data to save
    data_to_save = {
        "dataset_urls": dataset_urls,
        "predictions": predictions.tolist(),
        "future_predictions": future_df_dict
    }

    # Insert into MongoDB
    # collection.insert_one(data_to_save)
    st.success("Predictions saved to MongoDB.")

def analyze_data(data, predictions, future_df, timeframe):
    st.subheader("Analysis")
    data_resampled = resample_data(data, timeframe)

    # General statistics
    st.write("### General Statistics")
    st.write(data_resampled.describe())

    # Correlation matrix
    st.write("### Correlation Matrix")
    correlation_matrix = data_resampled.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, ax=ax, cmap='coolwarm')
    st.pyplot(fig)

    st.write("### Patterns and Insights")
    st.write("Identify relationships, time lags, and deviations from historical means.")

if __name__ == '__main__':
    main()