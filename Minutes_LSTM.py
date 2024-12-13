from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Flask app initialization
app = Flask(__name__)

# Constants
API_KEY = 'YOUR_API_KEY'
COINS = ['HMSTR', 'PEPE', 'ACT', 'SHIB', 'MATIC', 'MTL', 'DOGE', 'BTC', 'SOL', 'ONDO']
CURRENCY = 'USD'
PUSHBULLET_API_KEY = 'o.MfcM7XP7IAmiE55iWBqCUei81d48P6p7'
TIME_STEP = 12
PREDICTION_INTERVALS = 12
DAYS = 7

# Function to fetch data
def fetch_7_days_data(coin_id):
    all_data = []
    end_time = int(datetime.now().timestamp())

    for _ in range(DAYS):
        url = f"https://min-api.cryptocompare.com/data/v2/histominute?fsym={coin_id}&tsym={CURRENCY}&limit=1440&toTs={end_time}&api_key={API_KEY}"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch data for {coin_id}: {response.text}")

        data = response.json()
        if data.get('Response') == 'Success':
            records = data['Data']['Data']
            all_data.extend(records)
            end_time = records[0]['time'] - 1
        else:
            raise Exception(f"Error fetching data for {coin_id}: {data.get('Message', 'Unknown error')}")

    df = pd.DataFrame({
        "Time": [datetime.fromtimestamp(record['time']) for record in all_data],
        "Price": [record['close'] for record in all_data]
    })
    df.set_index("Time", inplace=True)
    return df.sort_index()

# Function to create the dataset
def create_dataset(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Function to build the model
def build_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    return model

# Route to trigger predictions
@app.route('/predict', methods=['GET'])
def predict():
    predictions = {}
    for coin_id in COINS:
        try:
            data = fetch_7_days_data(coin_id)
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data['Price'].values.reshape(-1, 1))
            X, y = create_dataset(scaled_data, TIME_STEP)
            X = X.reshape(X.shape[0], X.shape[1], 1)

            model = build_model((X.shape[1], 1))
            model.fit(X, y, epochs=10, batch_size=64, validation_split=0.2, verbose=0)

            # Predict future prices
            last_sequence = scaled_data[-TIME_STEP:]
            predicted_prices = []
            for _ in range(PREDICTION_INTERVALS):
                prediction = model.predict(last_sequence.reshape(1, TIME_STEP, 1))
                predicted_price = scaler.inverse_transform(prediction)[0][0]
                predicted_prices.append(predicted_price)
                last_sequence = np.append(last_sequence[1:], prediction, axis=0)

            # Calculate percentage changes
            current_price = data['Price'].iloc[-1]
            percentage_changes = [(pred - current_price) / current_price * 100 for pred in predicted_prices]

            predictions[coin_id] = {
                "current_price": current_price,
                "predicted_prices": predicted_prices,
                "percentage_changes": percentage_changes
            }
        except Exception as e:
            predictions[coin_id] = {"error": str(e)}
    
    return jsonify(predictions)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)