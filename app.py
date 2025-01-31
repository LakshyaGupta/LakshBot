import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import seaborn as sns
from datetime import datetime, timedelta
from pymongo import MongoClient
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

st.set_page_config(page_title="Stock Analysis & Prediction Dashboard", layout="wide")

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["stock_predictions"]
collection = db["predictions"]

# Password Protection Flag
password_correct = False

# Define the RL Environment for stock prediction
class PredictiveRLEnv:
    def __init__(self, data, clusters):
        self.data = data
        self.clusters = clusters
        self.current_step = 0
        self.n_steps = len(data) - 1
        
    def reset(self):
        self.current_step = 0
        return self.data[self.current_step]
    
    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.n_steps
        next_state = self.data[self.current_step] if not done else self.data[-1]
        reward = -np.abs(action - next_state).mean()  # Reward based on how close action is to next state
        return next_state, reward, done, {}

def password_protection():
    global password_correct
    password = st.sidebar.text_input("Enter password to access the dashboard", type="password")
    if password == "test":  # Replace with your desired password
        password_correct = True
        st.sidebar.success("Password correct! You now have access.")
        main_runner()
    elif password:
        st.sidebar.error("Incorrect password. Please try again.")

def download_data_from_fred(series_id, start_date, end_date):
    api_key = "cdde234b4a095daa82255f352e612845"  # Replace with your actual FRED API key
    api_url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json&observation_start={start_date}&observation_end={end_date}"
    response = requests.get(api_url)
    if response.status_code == 200:
        observations = response.json().get("observations", [])
        data = pd.DataFrame(observations)
        data = data.rename(columns={"date": "Date", "value": "Close"})
        # Select only necessary columns and convert
        data = data[['Date', 'Close']]  # Fix: Select only relevant columns
        data["Date"] = pd.to_datetime(data["Date"])
        data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
        data = data.dropna().set_index("Date")
        return data
    else:
        st.error("Failed to fetch data from FRED API. Check your API key or series ID.")
        return pd.DataFrame()

def combine_datasets(data_dict):
    # Concatenate all datasets into one DataFrame
    combined_data = pd.concat(data_dict.values(), axis=1, join="outer")
    # Set column names to dataset keys (each dataset has a single 'Close' column)
    combined_data.columns = data_dict.keys()
    return combined_data

def visualize_data(data, timeframe):
    data_resampled = resample_data(data, timeframe)
    st.line_chart(data_resampled)

def resample_data(data, timeframe):
    numeric_data = data.select_dtypes(include=[np.number])
    if timeframe == "Weekly":
        return numeric_data.resample('W').mean()
    elif timeframe == "Monthly":
        return numeric_data.resample('M').mean()
    elif timeframe == "Yearly":
        return numeric_data.resample('Y').mean()
    else:
        return numeric_data

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

            # LSTM Predictions
            lstm_predictions, future_df, backtested_lstm = predict_with_lstm(combined_data, prediction_years)
            
            # Reinforcement Learning Predictions
            rl_predictions, backtested_rl = predict_with_rl(combined_data)
            
            # Combine results
            backtested_combined = pd.concat([backtested_lstm.rename(columns={'Predicted': 'LSTM'}), 
                                             backtested_rl.rename(columns={'Predicted_RL': 'RL'})], axis=1)

            st.subheader("Backtested Results Comparison")
            st.line_chart(backtested_combined)

            st.subheader("LSTM Future Predictions")
            st.write(future_df)
            
            st.subheader("Reinforcement Learning Insights")
            st.write(f"RL Model Accuracy: {np.mean(rl_predictions['Accuracy'])*100:.2f}%")
            st.line_chart(rl_predictions[['Actual', 'Predicted']])

            save_to_mongodb(dataset_urls, lstm_predictions, future_df, rl_predictions)
            analyze_data(combined_data, lstm_predictions, future_df, timeframe)
        else:
            st.error("No data found for the selected dataset URLs and date range.")

def predict_with_rl(data, n_clusters=5, n_components=3):
    # Preprocessing
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(scaled_data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(reduced_data)

    # RL Environment
    env = DummyVecEnv([lambda: PredictiveRLEnv(reduced_data, clusters)])
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=1000)

    # Evaluation
    env_eval = PredictiveRLEnv(reduced_data, clusters)
    state = env_eval.reset()
    predictions = []
    actuals = []
    rewards = []
    
    for _ in range(len(reduced_data)-1):
        action, _ = model.predict(state)
        next_state, reward, done, _ = env_eval.step(action)
        predictions.append(action)
        actuals.append(next_state)
        rewards.append(reward)
        state = next_state
        if done: break

    # Inverse transformations
    predictions = pca.inverse_transform(np.array(predictions).squeeze())
    predictions = scaler.inverse_transform(predictions)
    actuals = pca.inverse_transform(np.array(actuals).squeeze())
    actuals = scaler.inverse_transform(actuals)
    
    # Create DataFrame
    index = data.index[1:len(predictions)+1]
    results = pd.DataFrame({
        'Actual': actuals.mean(axis=1),
        'Predicted': predictions.mean(axis=1),
        'Accuracy': [1 if r > -0.1 else 0 for r in rewards]
    }, index=index)
    
    return results, results[['Actual', 'Predicted']]

def predict_with_lstm(data, prediction_years):
    prediction_days = prediction_years * 365
    dataset = data.mean(axis=1).dropna().values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data_len = int(len(dataset) * 0.8)
    train_data = scaled_data[:train_data_len]
    
    # LSTM Model
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
    model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)

    # Predictions
    test_data = scaled_data[train_data_len-60:]
    x_test = [test_data[i-60:i, 0] for i in range(60, len(test_data))]
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_train.shape[1], 1))
    
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    
    # Future predictions
    last_60_days = scaled_data[-60:]
    future_predictions = []
    for _ in range(prediction_days):
        x_future = np.reshape(last_60_days, (1, 60, 1))
        pred_price = model.predict(x_future, verbose=0)[0, 0]
        future_predictions.append(pred_price)
        last_60_days = np.append(last_60_days[1:], [[pred_price]], axis=0)
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=prediction_days)
    future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Predicted Close'])

    # Backtesting
    backtested = pd.DataFrame({
        'Actual': scaler.inverse_transform(scaled_data[train_data_len:].reshape(-1, 1)).flatten(),
        'Predicted': predictions.flatten()
    }, index=data.index[train_data_len:train_data_len+len(predictions)])
    
    return predictions, future_df, backtested

def save_to_mongodb(datasets, lstm_pred, future_pred, rl_pred):
    record = {
        "timestamp": datetime.now(),
        "datasets": datasets,
        "lstm_predictions": lstm_pred.tolist(),
        "future_predictions": future_pred.to_dict(),
        "rl_predictions": rl_pred.to_dict()
    }
    collection.insert_one(record)

def analyze_data(data, lstm_pred, future_pred, timeframe):
    st.subheader("Statistical Analysis")
    st.write(data.describe())
    
    st.subheader("Correlation Matrix")
    corr = data.corr()
    sns.heatmap(corr, annot=True)
    st.pyplot()
    
    st.subheader("LSTM Prediction vs Actual")
    fig, ax = plt.subplots()
    ax.plot(data.index, data.mean(axis=1), label='Actual')
    ax.plot(lstm_pred.index, lstm_pred, label='Predicted')
    plt.legend()
    st.pyplot(fig)

if __name__ == '__main__':
    main()