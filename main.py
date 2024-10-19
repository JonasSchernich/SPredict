import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import os

# Step 1: Load historical S&P 500 data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Step 2: Prepare data for LSTM
def prepare_data(data):
    # Calculate the target variable (1 if the minimum price in the next 10 days is more than 2% below today's closing price)
    data['Target'] = ((data['Close'].shift(-10) * 0.98) > data['Close']).astype(int)

    # Prepare the data for LSTM
    features = data[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)

    X, y = [], []
    for i in range(100, len(scaled_features) - 10):
        X.append(scaled_features[i - 100:i])
        y.append(data['Target'].iloc[i])

    return np.array(X), np.array(y), scaler

# Step 3: Create and train the LSTM model
def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 4: Train the model
def main_train():
    # Load historical data
    data = load_data('^GSPC', '1970-01-01', '2022-03-01')

    # Prepare the data
    X, y, scaler = prepare_data(data)

    # Create the LSTM model
    model = create_model((X.shape[1], 1))

    # EarlyStopping for training
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    # Train the model
    model.fit(X, y, batch_size=32, epochs=50, validation_split=0.2, callbacks=[early_stopping])

    # Save the trained model
    model.save('lstm_model.h5')
    print("Model trained and saved as lstm_model.h5.")

# Step 5: Load the trained model
def load_trained_model():
    if os.path.exists('lstm_model.h5'):
        return load_model('lstm_model.h5')
    else:
        print("No trained model found. Please train the model first.")
        return None

# Step 6: Prediction function
def predict_next_ten_days(model, scaler, last_100_days):
    last_100_days_scaled = scaler.transform(last_100_days)
    last_100_days_scaled = last_100_days_scaled.reshape((1, last_100_days_scaled.shape[0], 1))
    probability = model.predict(last_100_days_scaled)[0][0]  # Get the predicted probability
    return probability  # Return the probability

def main_predict(date_input):
    # Load the trained model
    model = load_trained_model()

    if model is None:
        return

    # Define the start date (fixed at 01.01.1970) and format the input date
    start_date = '1970-01-01'
    end_date = datetime.datetime.strptime(date_input, '%d.%m.%Y').strftime('%Y-%m-%d')

    # Load the S&P 500 data from 01.01.1970 up to the given end_date
    data = load_data('^GSPC', start_date, end_date)

    # Ensure there are at least 100 days of data
    if len(data) < 100:
        print(f"Not enough data to make a prediction for {end_date} (need at least 100 days).")
        return False

    # Extract the last 100 days of closing prices before the specified end_date
    last_100_days = data['Close'].values[-100:].reshape(-1, 1)

    # Load the data again to get the scaler (must be the same scaler used during training)
    _, _, scaler = prepare_data(data)

    # Predict the probability that the market will fall at least 2%
    probability = predict_next_ten_days(model, scaler, last_100_days)
    print(f"With a probability of {probability * 100:.2f}%, the market will fall at least 2% within the next 10 days from {date_input}.")

if __name__ == '__main__':
    # Uncomment the next line if you want to train the model
    #main_train()  # To train the model

    # Example usage for prediction
    main_predict('10.03.2020')  # Specify the date to predict from
