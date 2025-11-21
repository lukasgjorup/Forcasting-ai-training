import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# -----------------------------
# CONFIG — CHANGE THESE PATHS
# -----------------------------
CSV_PATH = "formatted_userOutputData.csv"
MODEL_PATH = "lstm_energy_model.keras"
SCALER_PATH = "scalers.pkl"

def detect_first_household(csv_path):
    df = pd.read_csv(csv_path)
    house_id = df["LCLid"].unique()[0]
    print(f"Automatically detected household: {house_id}")
    return house_id


HOUSE_ID = HOUSE_ID = detect_first_household(CSV_PATH)

TIMESTEPS = 24
FUTURE_STEPS = 12

numeric_cols = ['energy(kWh/hh)', 'temperature']


# ----------------------------------
# Load model + scalers
# ----------------------------------
def load_model_and_scalers():
    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scalers = pickle.load(f)
    return model, scalers


# ----------------------------------
# Load CSV & return last 24 raw points
# ----------------------------------
def load_last_24_from_csv(csv_path, house_id):
    df = pd.read_csv(csv_path)

    # Convert to numeric
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=numeric_cols, inplace=True)

    # Get a single household
    house = df[df["LCLid"] == house_id].copy()

    if len(house) < 24:
        raise ValueError("ERROR: Household does not have 24 rows of data!")

    # Sort by time
    house = house.sort_values("datetime")

    # Extract last 24 points
    last_24 = house[numeric_cols].values[-24:]  # shape (24,2)
    return last_24


# ----------------------------------
# Automatic scaling of last 24 inputs
# ----------------------------------
def scale_last_24(last_24, scalers):
    last_24_scaled = last_24.copy()

    last_24_scaled[:, 0] = scalers['energy(kWh/hh)'] \
        .transform(last_24_scaled[:, 0].reshape(-1, 1)).flatten()

    last_24_scaled[:, 1] = scalers['temperature'] \
        .transform(last_24_scaled[:, 1].reshape(-1, 1)).flatten()

    return last_24_scaled



# ----------------------------------
# Auto-regressive multi-step forecast
# ----------------------------------
def recursive_predict(last_24_scaled, model, scalers):
    predictions_scaled = []
    input_window = last_24_scaled.copy()

    for step in range(FUTURE_STEPS):
        X = input_window.reshape(1, TIMESTEPS, len(numeric_cols))
        pred_scaled = model.predict(X)[0]  # shape (2,)

        predictions_scaled.append(pred_scaled)

        # Shift window & append prediction
        input_window = np.vstack([input_window[1:], pred_scaled])

    predictions_scaled = np.array(predictions_scaled)

    # Inverse transform both outputs
    pred_energy_real = scalers['energy(kWh/hh)'] \
        .inverse_transform(predictions_scaled[:, 0].reshape(-1, 1)).flatten()

    pred_temp_real = scalers['temperature'] \
        .inverse_transform(predictions_scaled[:, 1].reshape(-1, 1)).flatten()

    return pred_energy_real, pred_temp_real


# ----------------------------------
# MAIN
# ----------------------------------
if __name__ == "__main__":

    # Load model & scalers
    model, scalers = load_model_and_scalers()

    # Load CSV and get last 24 raw points
    last_24_raw = load_last_24_from_csv(CSV_PATH, HOUSE_ID)

    # Normalize values using scalers
    last_24_scaled = scale_last_24(last_24_raw, scalers)

    # Predict recursively
    energy_pred_12, temp_pred_12 = recursive_predict(last_24_scaled, model, scalers)

    print("\n🔮 Predicted next 12 energy values:")
    print(energy_pred_12)

    print("\n🌡️ Predicted next 12 temperature values:")
    print(temp_pred_12)
