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
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(all_values)
        scalers[col] = scaler
        for key in households:
            households[key][:, i] = scaler.transform(households[key][:, i].reshape(-1, 1)).flatten()
    with open("scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)
    return households, scalers


# 3️⃣ Create sequences
timesteps = 24
features = len(numeric_cols)


def create_sequences(data):
    x, y = [], []
    for i in range(len(data) - timesteps):
        x.append(data[i:i + timesteps])
        y.append(data[i + timesteps, 0])  # Predict next energy
    return np.array(x), np.array(y)


# 4️⃣ Stack multiple households into one dataset
def stackHouseholds(localHouseholds):
    all_X, all_y = [], []
    for key in localHouseholds:
        X_house, y_house = create_sequences(localHouseholds[key])
        all_X.append(X_house)
        all_y.append(y_house)
    z = np.vstack(all_X)
    p = np.concatenate(all_y)
    print("X shape:", z.shape, "y shape:", p.shape)
    return z, p


# 5️⃣ Prediction for one household
def predict_house(house, scaler, model):
    print("Predicting...")
    first_key = list(house.keys())[0]
    X_seq, y_seq = create_sequences(house[first_key])

    pred_scaled = model.predict(X_seq)
    y_real = scaler['energy(kWh/hh)'].inverse_transform(y_seq.reshape(-1, 1))
    pred_real = scaler['energy(kWh/hh)'].inverse_transform(pred_scaled)

    plt.figure(figsize=(10, 5))
    plt.plot(y_real, label='True')
    plt.plot(pred_real, label='Predicted')
    plt.title(f"Energy Prediction")
    plt.xlabel('Timestep')
    plt.ylabel('Energy (kWh/hh)')
    plt.legend()
    plt.show()


# 6️⃣ Predict household wrapper
def predictHousehold(dataPath="../../prediction_Data.csv", scalerPath="scalers.pkl", modelPath="lstm_energy_model.keras"):
    testHouseholds = LoadAndProcessCSV(dataPath)
    with open(scalerPath, "rb") as f:
        scalers = pickle.load(f)

    for idx, (house_id, arr) in enumerate(testHouseholds.items()):
        if idx > 0:
            print("WARNING! Multiple households detected, using only the first one.")
        arr[:, 0] = scalers['energy(kWh/hh)'].transform(arr[:, 0].reshape(-1, 1)).flatten()
        arr[:, 1] = scalers['temperature'].transform(arr[:, 1].reshape(-1, 1)).flatten()

    from tensorflow.keras.models import load_model
    model = load_model(modelPath)
    predict_house(testHouseholds, scalers, model)


# ✅ 7️⃣ Main block — only runs if executed directly
if __name__ == "__main__":
    print("Starting LSTM energy model training...")

    households = LoadAndProcessCSV("../../formatted_data_100.csv")
    households, scalers = makeScalerAndNormalizeData(households)
    X, Y = stackHouseholds(households)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(timesteps, features)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, Y, epochs=25, batch_size=32, validation_split=0.2)
    model.save("lstm_energy_model.keras")

    print("✅ Training finished, running test prediction...")
    predictHousehold()
