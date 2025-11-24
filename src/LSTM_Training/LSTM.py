import random

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pickle

numeric_cols = ['energy(kWh/hh)', 'temperature']

# 1️⃣ Load and process CSV
def LoadAndProcessCSV(path="../../formatted_data.csv"):
    df = pd.read_csv(path)
    print(df.columns.tolist())

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=numeric_cols, inplace=True)

    households = {}
    for house_id, group in df.groupby('LCLid'):
        group = group.sort_values('datetime')
        households[house_id] = group[numeric_cols].values
    return households


# 2️⃣ Normalize data
def makeScalerAndNormalizeData(households):
    scalers = {}

    for i, col in enumerate(numeric_cols):
        all_values = np.concatenate([house[:, i] for house in households.values()]).reshape(-1, 1)
        scaler = MinMaxScaler((0, 1))
        scaler.fit(all_values)
        scalers[col] = scaler

        # Apply scaler to each household
        for key in households:
            households[key][:, i] = scaler.transform(households[key][:, i].reshape(-1, 1)).flatten()

    with open("scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)

    return households, scalers


# 3️⃣ Create sequences (NOW output both variables!)
timesteps = 24
features = len(numeric_cols)

def create_sequences(data):
    X, Y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i + timesteps])
        Y.append(data[i + timesteps])   # returns BOTH: [energy, temperature]
    return np.array(X), np.array(Y)


# 4️⃣ Stack all households into one dataset
def stackHouseholds(localHouseholds):
    all_X, all_Y = [], []
    for key in localHouseholds:
        Xh, Yh = create_sequences(localHouseholds[key])
        all_X.append(Xh)
        all_Y.append(Yh)

    X = np.vstack(all_X)
    Y = np.vstack(all_Y)

    print("X shape:", X.shape, "Y shape:", Y.shape)
    return X, Y


# 5️⃣ Prediction for one household
def predict_house(households, scalers, model):
    print("Predicting...")

    first_key = list(households.keys())[0]
    data = households[first_key]

    X_seq, Y_seq = create_sequences(data)

    pred_scaled = model.predict(X_seq)

    # Split outputs
    pred_energy_scaled = pred_scaled[:, 0].reshape(-1, 1)
    pred_temp_scaled   = pred_scaled[:, 1].reshape(-1, 1)

    Y_energy_scaled = Y_seq[:, 0].reshape(-1, 1)
    Y_temp_scaled   = Y_seq[:, 1].reshape(-1, 1)

    # Inverse transform
    pred_energy = scalers['energy(kWh/hh)'].inverse_transform(pred_energy_scaled)
    real_energy = scalers['energy(kWh/hh)'].inverse_transform(Y_energy_scaled)

    pred_temp = scalers['temperature'].inverse_transform(pred_temp_scaled)
    real_temp = scalers['temperature'].inverse_transform(Y_temp_scaled)

    # Plot Energy
    plt.figure(figsize=(10, 5))
    plt.plot(real_energy, label="True Energy")
    plt.plot(pred_energy, label="Predicted Energy")
    plt.legend()
    plt.title("Energy Prediction")
    plt.xlabel("Timestep")
    plt.ylabel("Energy (kWh/hh)")
    plt.show()

    # Plot Temperature
    plt.figure(figsize=(10, 5))
    plt.plot(real_temp, label="True Temperature")
    plt.plot(pred_temp, label="Predicted Temperature")
    plt.legend()
    plt.title("Temperature Prediction")
    plt.xlabel("Timestep")
    plt.ylabel("Temperature")
    plt.show()


# 6️⃣ Prediction wrapper
def predictHousehold(
    dataPath="../../prediction_Data.csv",
    scalerPath="scalers.pkl",
    modelPath="lstm_energy_model.keras"
):
    testHouseholds = LoadAndProcessCSV(dataPath)

    with open(scalerPath, "rb") as f:
        scalers = pickle.load(f)

    from tensorflow.keras.models import load_model
    model = load_model(modelPath)



    evaluate_recursive_prediction(testHouseholds, scalers, model, steps=12)

import math
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_recursive_prediction(data_dict, scalers, model, timesteps=24, steps=12):
    results = {}

    for house_name, data in data_dict.items():
        if len(data) < timesteps + steps:
            print(f"Skipping {house_name}: not enough data.")
            continue

        # ---- 1) Build windows ----
        X, Y = [], []
        for i in range(len(data) - (timesteps + steps) + 1):
            X.append(data[i:i + timesteps])
            Y.append(data[i + timesteps:i + timesteps + steps])
        X = np.array(X)
        Y = np.array(Y)
        num_windows = len(X)

        # ---- 2) Normalize X ----
        X_scaled = np.zeros_like(X)
        for w in range(num_windows):
            win = X[w].copy()
            win[:, 0] = scalers['energy(kWh/hh)'].transform(win[:, 0].reshape(-1, 1)).flatten()
            win[:, 1] = scalers['temperature'].transform(win[:, 1].reshape(-1, 1)).flatten()
            X_scaled[w] = win

        # ---- 3) Recursive predictions ----
        preds_scaled = np.zeros((num_windows, steps, X.shape[2]))
        for w in range(num_windows):
            window = X_scaled[w].copy()
            for s in range(steps):
                X_input = window.reshape(1, timesteps, X.shape[2])
                pred = model.predict(X_input, verbose=0)[0]
                preds_scaled[w, s] = pred
                window = np.vstack([window[1:], pred])

        # ---- 4) Inverse transform predictions ----
        preds = np.zeros_like(preds_scaled)
        for w in range(num_windows):
            preds[w, :, 0] = scalers['energy(kWh/hh)'].inverse_transform(
                preds_scaled[w, :, 0].reshape(-1, 1)
            ).flatten()
            preds[w, :, 1] = scalers['temperature'].inverse_transform(
                preds_scaled[w, :, 1].reshape(-1, 1)
            ).flatten()

        # ---- 5) Compute metrics ----
        true = Y.copy()
        mae_energy, rmse_energy = [], []
        mae_temp, rmse_temp = [], []

        for w in range(num_windows):
            e_true, e_pred = true[w, :, 0], preds[w, :, 0]
            t_true, t_pred = true[w, :, 1], preds[w, :, 1]

            mae_energy.append(mean_absolute_error(e_true, e_pred))
            rmse_energy.append(math.sqrt(mean_squared_error(e_true, e_pred)))

            mae_temp.append(mean_absolute_error(t_true, t_pred))
            rmse_temp.append(math.sqrt(mean_squared_error(t_true, t_pred)))

        # ---- 6) Store results ----
        results[house_name] = {
            "predictions": preds,
            "true": true,
            "mae_energy": np.mean(mae_energy),
            "rmse_energy": np.mean(rmse_energy),
            "mae_temp": np.mean(mae_temp),
            "rmse_temp": np.mean(rmse_temp)
        }

        # ---- 7) Plot ONLY last window ----

        last_w = random.randint(0, num_windows-1)
        plt.figure(figsize=(10, 4))
        plt.plot(true[last_w, :, 0], label="True Energy")
        plt.plot(preds[last_w, :, 0], label="Pred Energy")
        plt.title(f"{house_name} — Energy (Last Window)")
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(true[last_w, :, 1], label="True Temp")
        plt.plot(preds[last_w, :, 1], label="Pred Temp")
        plt.title(f"{house_name} — Temperature (Last Window)")
        plt.legend()
        plt.show()

        print(f"\n📊 {house_name} — Recursive Forecast Metrics")
        print("----------------------------------------")
        print(f"Energy MAE: {results[house_name]['mae_energy']:.4f}")
        print(f"Energy RMSE: {results[house_name]['rmse_energy']:.4f}")
        print(f"Temp   MAE: {results[house_name]['mae_temp']:.4f}")
        print(f"Temp   RMSE: {results[house_name]['rmse_temp']:.4f}")


# 7️⃣ Main training script
if __name__ == "__main__":
    print("Starting dual-output LSTM training...")

    households = LoadAndProcessCSV("../../formatted_data_1200.csv")
    households, scalers = makeScalerAndNormalizeData(households)

    X, Y = stackHouseholds(households)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(timesteps, features)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(2)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, Y, epochs=20, batch_size=32, validation_split=0.2)

    model.save("lstm_energy_model.keras")

    print("Training done — running test prediction...")
    predictHousehold()
