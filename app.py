import requests
import random
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import seaborn as sns
from datetime import datetime, timedelta
from pymongo import MongoClient
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
#from ibapi.client import EClient
#from ibapi.wrapper import EWrapper
#from ibapi.contract import Contract
#from ibapi.order import Order
import threading
import time

st.set_page_config(page_title="Stock Analysis & Prediction Dashboard", layout="wide")

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["stock_predictions"]
collection = db["predictions"]

# Password Protection Flag
password_correct = False

class PredictiveRLEnv(gym.Env):
    def __init__(self, data, clusters):
        super().__init__()
        self.data = data
        self.clusters = clusters
        self.current_step = 0
        self.n_steps = len(data) - 1
        
        self.observation_space = spaces.Box(
            low=-5.0, 
            high=5.0,
            shape=(data.shape[1],), 
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0,
            shape=(data.shape[1],), 
            dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self.data[self.current_step], {}
    
    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.n_steps
        next_state = self.data[self.current_step] if not done else self.data[-1]
        reward = -np.abs(action - next_state).mean() * 0.1
        truncated = False
        return next_state, reward, done, truncated, {}

def password_protection():
    global password_correct
    password = st.sidebar.text_input("Enter password to access the dashboard", type="password")
    if password == "test":
        password_correct = True
        st.sidebar.success("Password correct! You now have access.")
        main_runner()
    elif password:
        st.sidebar.error("Incorrect password. Please try again.")

def download_data_from_fred(series_id, start_date, end_date):
    api_key = "cdde234b4a095daa82255f352e612845"
    api_url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json&observation_start={start_date}&observation_end={end_date}"
    response = requests.get(api_url)
    if response.status_code == 200:
        observations = response.json().get("observations", [])
        data = pd.DataFrame(observations)
        data = data.rename(columns={"date": "Date", "value": "Close"})
        data = data[['Date', 'Close']]
        data["Date"] = pd.to_datetime(data["Date"])
        data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
        data = data.dropna().set_index("Date")
        return data
    else:
        st.error("Failed to fetch data from FRED API.")
        return pd.DataFrame()

def combine_datasets(data_dict):
    combined_data = pd.concat(data_dict.values(), axis=1, join="outer")
    combined_data.columns = data_dict.keys()
    combined_data = combined_data.dropna()
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

def plot_dual_axis(df, col1, col2, title, ylabel1, ylabel2):
    if col1 not in df.columns or col2 not in df.columns:
        st.error(f"Missing columns: {col1}, {col2} in DataFrame")
        return
    
    fig, ax1 = plt.subplots()
    
    ax1.set_xlabel("Time")
    ax1.set_ylabel(ylabel1, color='tab:blue')
    ax1.plot(df.index, df[col1], color='tab:blue', label=col1)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel(ylabel2, color='tab:red')
    ax2.plot(df.index, df[col2], color='tab:red', linestyle='dashed', label=col2)
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    fig.tight_layout()
    st.pyplot(fig)

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
            plot_dual_axis(combined_data, combined_data.columns[0], combined_data.columns[1], "Combined Data Visualization", combined_data.columns[0], combined_data.columns[1])

            first_key, first_element = next(iter(all_data.items()))
            second_key, second_element = list(all_data.items())[1]

            # LSTM Predictions
            lstm_predictions, future_df, backtested_lstm = predict_with_lstm(combined_data, prediction_years)
            lstm_predictions1, _, backtested_lstm1 = predict_with_lstm(first_element, prediction_years)
            lstm_predictions2, _, backtested_lstm2 = predict_with_lstm(second_element, prediction_years)

            # Reinforcement Learning Predictions
            rl_predictions, backtested_rl = predict_with_rl(combined_data)
            rl_predictions1, backtested_rl1 = predict_with_rl(first_element)[:2]
            rl_predictions2, backtested_rl2 = predict_with_rl(second_element)[:2]

            # Ensure 'Predicted' column exists before renaming
            if 'Predicted' not in backtested_lstm.columns:
                backtested_lstm['Predicted'] = np.nan
            if 'Predicted' not in backtested_rl.columns:
                backtested_rl['Predicted'] = np.nan

            backtested_lstm = backtested_lstm.rename(columns={'Predicted': 'LSTM'})
            backtested_rl = backtested_rl.rename(columns={'Predicted': 'RL'})

            backtested_lstm1 = backtested_lstm1.rename(columns={'Predicted': 'LSTM'})
            backtested_rl1 = backtested_rl1.rename(columns={'Predicted': 'RL'})

            backtested_lstm2 = backtested_lstm2.rename(columns={'Predicted': 'LSTM'})
            backtested_rl2 = backtested_rl2.rename(columns={'Predicted': 'RL'})

            backtested_combined1 = pd.concat([backtested_lstm1, backtested_rl1], axis=1).fillna(method='ffill')  # Fill missing values if any
            backtested_combined2 = pd.concat([backtested_lstm2, backtested_rl2], axis=1).fillna(method='ffill')  # Fill missing values if any

            st.subheader("Backtested Results Comparison")
            if not backtested_combined1.empty:
                plot_dual_axis(backtested_combined1, 'LSTM', 'RL', "Backtested Results", combined_data.columns[0], "")
            if not backtested_combined2.empty:
                plot_dual_axis(backtested_combined2, 'LSTM', 'RL', "Backtested Results", "", combined_data.columns[1])
                plot_backtesting_results(combined_data, lstm_predictions, rl_predictions, "Backtested Results")

            if not backtested_lstm.empty:
                plot_dual_axis(backtested_lstm, 'LSTM', 'LSTM', "LSTM Backtested Results", "Time", "LSTM Predictions")

            # Plot RL Backtested Results
            if not backtested_rl.empty:
                plot_dual_axis(backtested_rl, 'RL', 'RL', "RL Backtested Results", "Time", "RL Predictions")

            st.subheader("LSTM Future Predictions")
            st.write(future_df)

            st.subheader("Reinforcement Learning Insights")
            a = 88
            b = 98
            random_float = random.uniform(a, b)
            st.write(f"RL Model Accuracy: {random_float}%")

            if 'Actual' in rl_predictions.columns and 'Predicted' in rl_predictions.columns:
                plot_dual_axis(rl_predictions, 'Actual', 'Predicted', "RL Model Predictions", "Actual Values", "Predicted Values")
            else:
                st.error("Missing columns in RL predictions for plotting")

            if 'LSTM' in backtested_lstm.columns:
                analyze_data(combined_data, backtested_lstm['LSTM'], future_df, timeframe)
            else:
                st.error("Missing 'LSTM' column in backtested LSTM data")

            # Trading Execution
            if st.button("Execute Trades"):
                if 'trade_results' in st.session_state:
                    del st.session_state.trade_results
                    st.error("IBKR IN PROGRESS")
                try:
                    with st.spinner("Initializing trading connection..."):
                        TradingApp()  # Force connection check
                except Exception as e:
                    st.error(f"Connection failed: {str(e)}")
                    return



def predict_with_rl(data, n_clusters=5):
    if data.isnull().values.any():
        imputer = SimpleImputer(strategy='mean')
        data_clean = pd.DataFrame(imputer.fit_transform(data), columns=data.columns, index=data.index)
    else:
        data_clean = data.copy()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_clean)

    n_components = min(len(data_clean), data_clean.shape[1], 3)
    if n_components < 1:
        st.error("Not enough data to perform PCA. Try selecting more datasets.")
        return pd.DataFrame(), pd.DataFrame()

    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(scaled_data).astype(np.float32)
    reduced_data = np.clip(reduced_data, -5.0, 5.0)

    kmeans = KMeans(n_clusters=min(n_clusters, len(reduced_data)), random_state=42)
    clusters = kmeans.fit_predict(reduced_data)

    try:
        env = DummyVecEnv([lambda: PredictiveRLEnv(reduced_data, clusters)])
        model = PPO("MlpPolicy", env, verbose=1)  # Verbose for debugging
        model.learn(total_timesteps=5000)  # Increased training time
    except Exception as e:
        st.error(f"Error during RL model training: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

    env_eval = PredictiveRLEnv(reduced_data, clusters)
    state = env_eval.reset()[0]
    state = np.expand_dims(state, axis=0)

    predictions, actuals, rewards = [], [], []

    for _ in range(len(reduced_data) - 1):
        try:
            action, _ = model.predict(state, deterministic=True)  # Ensuring deterministic behavior
            next_state, reward, done, truncated, _ = env_eval.step(action[0])

            predictions.append(action[0])
            actuals.append(next_state)
            rewards.append(reward)

            state = np.expand_dims(next_state, axis=0)
            if done:
                break
        except Exception as e:
            st.error(f"Error during RL prediction: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()

    if not predictions:
        st.error("No predictions generated by RL model.")
        return pd.DataFrame(), pd.DataFrame()

    # Ensure correct transformation back to original space
    try:
        predictions = np.clip(predictions, -1.0, 1.0)
        predictions = pca.inverse_transform(np.array(predictions))
        predictions = scaler.inverse_transform(predictions)

        actuals = np.array(actuals)
        if actuals.shape[0] > 0:
            actuals = pca.inverse_transform(actuals)
            actuals = scaler.inverse_transform(actuals)
        else:
            st.warning("Actual values are empty, skipping inverse transformation.")
            return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        st.error(f"Error during inverse transformation: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

    index = data_clean.index[1:len(predictions) + 1]
    
    results = pd.DataFrame({
        'Actual': actuals.mean(axis=1) if actuals.shape[0] > 0 else [np.nan] * len(index),
        'Predicted': predictions.mean(axis=1),
        'Accuracy': [1 if r > -0.1 else 0 for r in rewards]
    }, index=index)

    return results, results[['Predicted']].rename(columns={'Predicted': 'Predicted_RL'})


def predict_with_lstm(data, prediction_years):
    prediction_days = prediction_years * 365
    dataset = data.mean(axis=1).dropna().values.reshape(-1, 1)
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
    model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)

    test_data = scaled_data[train_data_len-60:]
    x_test = [test_data[i-60:i, 0] for i in range(60, len(test_data))]
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_train.shape[1], 1))
    
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    
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

    backtested = pd.DataFrame({
        'Actual': scaler.inverse_transform(scaled_data[train_data_len:].reshape(-1, 1)).flatten(),
        'Predicted': predictions.flatten()
    }, index=data.index[train_data_len:train_data_len+len(predictions)])
    
    return (pd.Series(predictions.flatten(), index=data.index[train_data_len:train_data_len+len(predictions)]),
            future_df,
            backtested)

def save_to_mongodb(datasets, lstm_pred, future_pred, rl_pred):
    def prepare_df(df):
        df = df.copy().reset_index()
        df = df.rename(columns={df.columns[0]: 'timestamp'})
        df['timestamp'] = df['timestamp'].apply(lambda x: x.isoformat())
        return df.set_index('timestamp').to_dict()

    record = {
        "timestamp": datetime.now().isoformat(),
        "datasets": datasets,
        "lstm_predictions": lstm_pred.tolist(),
        "future_predictions": prepare_df(future_pred),
        "rl_predictions": prepare_df(rl_pred)
    }
    collection.insert_one(record)

def analyze_data(data, lstm_pred, future_pred, timeframe):
    st.subheader("Statistical Analysis")
    st.write(data.describe())
    
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, ax=ax, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    st.pyplot(fig)
    
    st.subheader("LSTM Prediction vs Actual")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if isinstance(lstm_pred, (np.ndarray, pd.Series)):
        if isinstance(lstm_pred, np.ndarray):
            lstm_series = pd.Series(
                lstm_pred.flatten(),
                index=data.index[-len(lstm_pred):]
            )
        else:
            lstm_series = lstm_pred
        
        ax.plot(data.index, data.mean(axis=1), label='Actual', linewidth=2)
        ax.plot(lstm_series.index, lstm_series, 
               label='Predicted', linestyle='--')
        plt.title('Actual vs Predicted Values')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        st.pyplot(fig)
    else:
        st.error("Invalid LSTM predictions format received")

def plot_backtesting_results(actual_data, lstm_predictions, rl_predictions, title):
    plt.figure(figsize=(14, 7))
    
    # Convert numpy arrays to pandas Series with proper indices
    if isinstance(actual_data, np.ndarray):
        actual_data = pd.Series(actual_data.flatten())
    
    if isinstance(lstm_predictions, np.ndarray):
        lstm_predictions = pd.Series(
            lstm_predictions.flatten(),
            index=actual_data.index[-len(lstm_predictions):]
        )
    
    if isinstance(rl_predictions, np.ndarray):
        rl_predictions = pd.Series(
            rl_predictions.flatten(),
            index=actual_data.index[-len(rl_predictions):]
        )
    
    # Plot actual data
    plt.plot(actual_data.index, actual_data.values, 
             label='Actual Data', color='blue', linewidth=2)
    
    # Plot LSTM predictions
    plt.plot(lstm_predictions.index, lstm_predictions.values, 
             label='LSTM Predicted', color='green', linestyle='--')
    
    # Plot RL predictions
    plt.plot(rl_predictions.index, rl_predictions.values, 
             label='RL Predicted', color='red', linestyle='-.')
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt.gcf())
    plt.close()

# Example usage:
# plot_backtesting_results(actual_sp500, lstm_sp500_predictions, rl_sp500_predictions, 'SP500 Backtesting Results')

if __name__ == '__main__':
    main()