import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from datetime import datetime, timedelta
import os

# Globale Definition der Features
features = ['Close', 'Bullish', 'Neutral', 'Bearish', 'Yield']


def load_sentiment_data(file_path):
    sentiment_data = pd.read_excel(file_path, engine='xlrd', skiprows=3)
    sentiment_data = sentiment_data.drop(sentiment_data.index[:27])
    sentiment_data = sentiment_data[['Date', 'Bullish', 'Neutral', 'Bearish']]
    sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date'], errors='coerce')
    sentiment_data = sentiment_data.dropna(subset=['Date'])
    sentiment_data.set_index('Date', inplace=True)
    return sentiment_data


def load_sp500_data(start_date, end_date):
    sp500_data = yf.download('^GSPC', start=start_date, end=end_date)
    sp500_data = sp500_data[['Close']]
    sp500_data.index = pd.to_datetime(sp500_data.index)
    sp500_data.index.name = 'Date'
    return sp500_data


def load_treasury_data(file_path):
    treasury_data = pd.read_excel(file_path)
    treasury_data['Date'] = pd.to_datetime(treasury_data['Date'])
    treasury_data.set_index('Date', inplace=True)
    treasury_data['Yield'] = treasury_data['Yield'].interpolate(method='linear')
    return treasury_data


def merge_data(sentiment_data, sp500_data, treasury_data):
    merged_data = pd.merge_asof(sp500_data, sentiment_data, left_index=True, right_index=True, direction='backward')
    merged_data = pd.merge_asof(merged_data, treasury_data, left_index=True, right_index=True, direction='nearest')
    merged_data[['Bullish', 'Neutral', 'Bearish']] = merged_data[['Bullish', 'Neutral', 'Bearish']].interpolate(
        method='linear')
    merged_data[['Bullish', 'Neutral', 'Bearish']] = merged_data[['Bullish', 'Neutral', 'Bearish']].ffill().bfill()
    return merged_data


def create_target(data, window=252):  # 252 Handelstage in einem Jahr
    data['Return'] = data['Close'].pct_change(window)
    data['Target'] = (data['Return'] > 0.1).astype(int)  # 1 wenn Rendite > 10%, sonst 0
    data = data.dropna()
    return data


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[features].values[i:(i + seq_length)])
        y.append(data['Target'].values[i + seq_length])
    return np.array(X), np.array(y)


def build_model(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def save_model_and_scaler(model, scaler):
    save_model(model, 'sp500_prediction_model.h5')
    np.save('scaler.npy', scaler.scale_)
    np.save('scaler_min.npy', scaler.min_)
    np.save('scaler_data_range.npy', scaler.data_range_)
    np.save('features.npy', features)


def load_trained_model():
    global features
    if not all(os.path.exists(f) for f in
               ['sp500_prediction_model.h5', 'scaler.npy', 'scaler_min.npy', 'scaler_data_range.npy', 'features.npy']):
        print("Erforderliche Modelldateien fehlen. Ein neues Modell muss trainiert werden.")
        return None, None

    try:
        model = load_model('sp500_prediction_model.h5')
        scaler = MinMaxScaler()
        scaler.scale_ = np.load('scaler.npy')
        scaler.min_ = np.load('scaler_min.npy')
        scaler.data_range_ = np.load('scaler_data_range.npy')
        features = np.load('features.npy', allow_pickle=True).tolist()
        return model, scaler
    except Exception as e:
        print(f"Fehler beim Laden des Modells: {e}")
        return None, None


def train_model(merged_data):
    global scaler
    scaler = MinMaxScaler()

    merged_data[features] = scaler.fit_transform(merged_data[features])
    merged_data = create_target(merged_data)

    seq_length = 60
    X, y = create_sequences(merged_data, seq_length)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model((seq_length, len(features)))
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        verbose=1
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    save_model_and_scaler(model, scaler)
    merged_data.to_csv('merged_data.csv')
    return model, scaler


def predict_for_date(model, scaler, merged_data, input_date):
    input_date = pd.to_datetime(input_date)
    if input_date > merged_data.index[-1]:
        print(f"Warnung: Vorhersage für ein Datum in der Zukunft ({input_date})")
        input_date = merged_data.index[-1]

    seq_length = 60
    end_idx = merged_data.index.get_loc(input_date)
    start_idx = max(0, end_idx - seq_length + 1)
    sequence = merged_data[features].iloc[start_idx:end_idx + 1].values

    if len(sequence) < seq_length:
        print(f"Nicht genügend Daten für eine Vorhersage. Benötige mindestens {seq_length} Tage.")
        return None

    sequence = sequence[-seq_length:]  # Ensure we have exactly seq_length days
    sequence = scaler.transform(sequence)
    sequence = np.expand_dims(sequence, axis=0)

    prediction = model.predict(sequence)[0][0]
    return prediction


def main():
    sentiment_file_path = 'fixtures/sentiment.xls'
    treasury_file_path = 'fixtures/DGS10.xls'
    start_date = '1988-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')

    model, scaler = load_trained_model()
    if model is None or scaler is None:
        print("Training new model...")
        sentiment_data = load_sentiment_data(sentiment_file_path)
        sp500_data = load_sp500_data(start_date, end_date)
        treasury_data = load_treasury_data(treasury_file_path)
        merged_data = merge_data(sentiment_data, sp500_data, treasury_data)
        model, scaler = train_model(merged_data)
    else:
        print("Existing model loaded successfully.")
        merged_data = pd.read_csv('merged_data.csv', index_col='Date', parse_dates=True)

    while True:
        input_date = input("Geben Sie ein Datum ein (YYYY-MM-DD) oder 'q' zum Beenden: ")
        if input_date.lower() == 'q':
            break

        try:
            prediction = predict_for_date(model, scaler, merged_data, input_date)
            if prediction is not None:
                print(f"\nVorhersage für {input_date}:")
                print(
                    f"Mit einer Wahrscheinlichkeit von {prediction:.2%} wird der S&P 500 in einem Jahr mehr als 10% über dem Kurs vom {input_date} liegen.")

                # Aktuelle Daten abrufen
                input_date_obj = datetime.strptime(input_date, "%Y-%m-%d")
                current_data = yf.download('^GSPC', start=input_date_obj, end=input_date_obj + timedelta(days=1))
                if not current_data.empty:
                    input_close = current_data['Close'].iloc[0]
                    print(f"\nS&P 500 Stand am {input_date}: {input_close:.2f}")
                    target_price = input_close * 1.1
                    print(f"10% Zielkurs: {target_price:.2f}")
                else:
                    print(f"\nKeine Daten für {input_date} verfügbar.")
        except ValueError:
            print("Ungültiges Datumsformat. Bitte verwenden Sie YYYY-MM-DD.")


if __name__ == '__main__':
    main() 