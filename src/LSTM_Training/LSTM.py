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

    # Normalize each household with saved scalers
    for i, (house_id, arr) in enumerate(testHouseholds.items()):
        if i > 0:
            print("WARNING: multiple households detected, using only the first one.")

        arr[:, 0] = scalers['energy(kWh/hh)'].transform(arr[:, 0].reshape(-1, 1)).flatten()
        arr[:, 1] = scalers['temperature'].transform(arr[:, 1].reshape(-1, 1)).flatten()

    from tensorflow.keras.models import load_model
    model = load_model(modelPath)

    first_key = list(testHouseholds.keys())[0]
    data = testHouseholds[first_key]

    evaluate_recursive_prediction(data, scalers, model, steps=12)

import math
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_recursive_prediction(data, scalers, model, steps=12):
    """
    Evaluates model by recursively predicting next N timesteps
    and comparing to ground truth.
    """

    if len(data) < timesteps + steps:
        raise ValueError(f"Not enough data to evaluate {steps} steps ahead.")

    # Split last window + true label sequence
    input_window = data[-(timesteps + steps):-steps]   # 24 inputs
    true_future = data[-steps:]                        # next 12 true labels

    # Normalize input window
    window_scaled = input_window.copy()
    window_scaled[:, 0] = scalers['energy(kWh/hh)'].transform(window_scaled[:, 0].reshape(-1, 1)).flatten()
    window_scaled[:, 1] = scalers['temperature'].transform(window_scaled[:, 1].reshape(-1, 1)).flatten()

    predictions_scaled = []

    # Recursive forecasting
    window = window_scaled.copy()

    for i in range(steps):
        X = window.reshape(1, timesteps, len(numeric_cols))
        pred_scaled = model.predict(X)[0]  # [energy, temp]

        predictions_scaled.append(pred_scaled)

        # shift window
        window = np.vstack([window[1:], pred_scaled])

    predictions_scaled = np.array(predictions_scaled)

    # Inverse-transform predictions
    pred_energy = scalers['energy(kWh/hh)'].inverse_transform(predictions_scaled[:, 0].reshape(-1, 1)).flatten()
    pred_temp   = scalers['temperature'].inverse_transform(predictions_scaled[:, 1].reshape(-1, 1)).flatten()

    # Extract real targets
    true_energy = true_future[:, 0]
    true_temp   = true_future[:, 1]

    # Compute errors
    mae_energy = mean_absolute_error(true_energy, pred_energy)
    rmse_energy = math.sqrt(mean_squared_error(true_energy, pred_energy))

    mae_temp = mean_absolute_error(true_temp, pred_temp)
    rmse_temp = math.sqrt(mean_squared_error(true_temp, pred_temp))

    print("\n📊 Recursive Forecast Evaluation")
    print("--------------------------------")
    print(f"Energy   MAE:  {mae_energy:.4f}")
    print(f"Energy   RMSE: {rmse_energy:.4f}")
    print(f"Temp     MAE:  {mae_temp:.4f}")
    print(f"Temp     RMSE: {rmse_temp:.4f}")

    # Plot energy
    plt.figure(figsize=(10, 4))
    plt.plot(true_energy, label="True Energy")
    plt.plot(pred_energy, label="Predicted Energy")
    plt.title("12-Step Recursive Prediction — Energy")
    plt.xlabel("Timestep")
    plt.ylabel("kWh/hh")
    plt.legend()
    plt.show()

    # Plot temperature
    plt.figure(figsize=(10, 4))
    plt.plot(true_temp, label="True Temperature")
    plt.plot(pred_temp, label="Predicted Temperature")
    plt.title("12-Step Recursive Prediction — Temperature")
    plt.xlabel("Timestep")
    plt.ylabel("°C")
    plt.legend()
    plt.show()


# 7️⃣ Main training script
if __name__ == "__main__":
    print("Starting dual-output LSTM training...")

    households = LoadAndProcessCSV("../../formatted_data_100.csv")
    households, scalers = makeScalerAndNormalizeData(households)

    X, Y = stackHouseholds(households)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(timesteps, features)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(2)       # TWO OUTPUTS
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, Y, epochs=25, batch_size=32, validation_split=0.2)

    model.save("lstm_energy_model.keras")

    print("Training done — running test prediction...")
    predictHousehold()
