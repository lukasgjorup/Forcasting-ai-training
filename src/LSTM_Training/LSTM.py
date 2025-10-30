import pandas as pd  # For reading CSVs and handling tabular data
import numpy as np  # For numerical operations and arrays
from tensorflow.keras.models import Sequential  # To create a sequential neural network
from tensorflow.keras.layers import LSTM, Dense  # LSTM for sequences, Dense for output layer
from sklearn.preprocessing import MinMaxScaler  # For normalizing feature values
import matplotlib.pyplot as plt  # For plotting predictions vs true values
from tensorflow.keras.layers import Dropout
import pickle

numeric_cols = ['energy(kWh/hh)', 'temperature']


def LoadAndProcessCSV(path="../../formatted_data.csv"):
    df = pd.read_csv(path)  # Load the CSV into a Pandas DataFrame
    print(df.columns.tolist())  # Print column names to verify they are correct
  # Columns we care about for model
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')  # Convert to numeric, invalid values become NaN
    df.dropna(subset=numeric_cols, inplace=True)  # Drop rows with missing values in these columns

    # 2️⃣ Group by household
    households = {}  # Dictionary to store data for each household
    for house_id, group in df.groupby('LCLid'):  # Group data by household ID
        group = group.sort_values('datetime')  # Sort each household's data chronologically
        households[house_id] = group[numeric_cols].values  # Save the numeric values as a NumPy array
    return households


households = LoadAndProcessCSV()

def makeScalerAndNormalizeData(households):
    # 3️⃣ Normalize features
    scalers = {}  # Store one scaler per feature
    for i, col in enumerate(numeric_cols):
        # Combine all values of this feature across all households for consistent scaling
        all_values = np.concatenate([house[:, i] for house in households.values()]).reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))  # Scale between 0 and 1
        scaler.fit(all_values)  # Fit scaler on all data for this feature
        scalers[col] = scaler  # Store the scaler
        for key in households:
            # Transform each household's data for this feature
            households[key][:, i] = scaler.transform(households[key][:, i].reshape(-1, 1)).flatten()
    with open("scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)
    return households, scalers

households,scalers = makeScalerAndNormalizeData(households)


# 4️⃣ Create sequences
timesteps = 24  # Use past 24 timesteps (e.g., hours) to predict next energy
features = len(numeric_cols)  # Number of input features (energy, temp, humidity)


def create_sequences(data):
    x, y = [], []  # Initialize input/output arrays
    for i in range(len(data) - timesteps):
        x.append(data[i:i + timesteps])  # Sequence of past timesteps
        y.append(data[i + timesteps, 0])  # Target is the next energy value (first column)
    return np.array(x), np.array(y)  # Convert lists to NumPy arrays

def stackHouseholds(localHouseholds):

    all_X, all_y = [], []  # Collect sequences for all households
    for key in localHouseholds:
        X_house, y_house = create_sequences(localHouseholds[key])  # Generate sequences
        all_X.append(X_house)  # Add to input list
        all_y.append(y_house)  # Add to output list

    z = np.vstack(all_X)  # Stack all households' sequences together
    p = np.concatenate(all_y)  # Combine all targets
    print("X shape:", z.shape, "y shape:", p.shape)  # Print shapes for verification
    return z, p

X,Y = stackHouseholds(households)

# 5️⃣ Build & train LSTM
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(timesteps, features)),
    Dropout(0.2),
    LSTM(32,return_sequences=True),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')  # Compile with Adam optimizer and MSE loss
model.fit(X, Y, epochs=10, batch_size=32, validation_split=0.2)  # Train model

model.save("lstm_energy_model.keras")


# 6️⃣ Predict per household
def predict_house(house,scaler):
    print("Predicting")
    #data = house[]  # Select household data
    #print(data)
    X_seq,y_seq = stackHouseholds(house) # Create sequences
    print(X_seq, "X")
    print(y_seq, "y")
    pred_scaled = model.predict(X_seq)  # Predict on normalized data
    # Convert back to real-world values
    y_real = scaler['energy(kWh/hh)'].inverse_transform(y_seq.reshape(-1, 1))
    print(pred_scaled, "pred_scaled")
    pred_last = pred_scaled[:, -1, :]
    pred_real = scaler['energy(kWh/hh)'].inverse_transform(pred_last)
    print(pred_real, "x_pred")
    print(y_real, "y real")

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(y_real, label='True')  # True energy values
    plt.plot(pred_real, label='Predicted')  # Predicted energy values
    plt.title(f"Energy Prediction")
    plt.xlabel('Timestep')
    plt.ylabel('Energy (kWh/hh)')
    plt.legend()
    plt.show()


def predictHousehold(dataPath = "../../prediction_Data.csv",scalerPath = "scalers.pkl"):
    testHouseholds = LoadAndProcessCSV(dataPath)
    with open(scalerPath, "rb") as f:
        scalers = pickle.load(f)

    for idx, (house_id, arr) in enumerate(testHouseholds.items()):
        if idx > 0:
            print("WARNING! User has sent more than one household")

        # Scale both columns
        arr[:, 0] = scalers['energy(kWh/hh)'].transform(arr[:, 0].reshape(-1, 1)).flatten()
        arr[:, 1] = scalers['temperature'].transform(arr[:, 1].reshape(-1, 1)).flatten()

    print(list(testHouseholds.keys()))

    predict_house(testHouseholds,scalers)  # Predict for the first household

predictHousehold()
