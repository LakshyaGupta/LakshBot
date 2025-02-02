import requests
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
import gym
from gym import spaces

st.set_page_config(page_title="Stock Analysis & Prediction Dashboard", layout="wide")

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["stock_predictions"]
collection = db["predictions"]

# Password Protection Flag
password_correct = False

class PredictiveRLEnv(gym.Env):
    def __init__(self, data, clusters):
        super(PredictiveRLEnv, self).__init__()
        self.data = data
        self.clusters = clusters
        self.current_step = 0
        self.n_steps = len(data) - 1
        
        # Define bounded action and observation spaces
        self.observation_space = spaces.Box(
            low=-5.0, 
            high=5.0,  # Adjusted for standardized PCA data
            shape=(data.shape[1],), 
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0,  # Bounded action space
            shape=(data.shape[1],), 
            dtype=np.float32
        )
        
    def reset(self):
        self.current_step = 0
        return self.data[self.current_step]
    
    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.n_steps
        next_state = self.data[self.current_step] if not done else self.data[-1]
        
        # Scale reward calculation to bounded action space
        reward = -np.abs(action - next_state).mean() * 0.1  # Scaled reward
        return next_state, reward, done, {}

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
            
            analyze_data(combined_data, backtested_lstm['Predicted'], future_df, timeframe)
        else:
            st.error("No data found for the selected dataset URLs and date range.")

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
    reduced_data = pca.fit_transform(scaled_data)
    
    # Clip PCA values to ensure observation space bounds
    reduced_data = np.clip(reduced_data, -5.0, 5.0)
    
    kmeans = KMeans(n_clusters=min(n_clusters, len(reduced_data)), random_state=42)
    clusters = kmeans.fit_predict(reduced_data)

    env = DummyVecEnv([lambda: PredictiveRLEnv(reduced_data, clusters)])
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=1000)

    env_eval = PredictiveRLEnv(reduced_data, clusters)
    state = env_eval.reset()
    predictions, actuals, rewards = [], [], []
    
    for _ in range(len(reduced_data)-1):
        action, _ = model.predict(state)
        next_state, reward, done, _ = env_eval.step(action)
        predictions.append(action)
        actuals.append(next_state)
        rewards.append(reward)
        state = next_state
        if done:
            break

    # Clip predictions to action space bounds before inverse transforms
    predictions = np.clip(predictions, -1.0, 1.0)
    predictions = pca.inverse_transform(np.array(predictions).squeeze())
    predictions = scaler.inverse_transform(predictions)
    actuals = pca.inverse_transform(np.array(actuals).squeeze())
    actuals = scaler.inverse_transform(actuals)
    
    index = data_clean.index[1:len(predictions)+1]
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
    
    return predictions, future_df, backtested

def save_to_mongodb(datasets, lstm_pred, future_pred, rl_pred):
    def prepare_df(df):
        df = df.copy().reset_index()
        # Rename index column to generic 'timestamp'
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

# Previous code remains identical until analyze_data function

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
    
    # Handle numpy array input
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

if __name__ == '__main__':
    main()