import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

# Import your functions here
from src.LSTM_Training.LSTM import (
    LoadAndProcessCSV,
    makeScalerAndNormalizeData,
    create_sequences,
    stackHouseholds,
    numeric_cols,
)

def make_fake_csv(tmp_path):
    """
       Creates a small fake CSV file.

       why is it made:
       - Ensures LoadAndProcessCSV can read a CSV file.
       - Avoids dependency on real datasets.
       - Allows predictable testing of CSV loading and grouping.
       """
    data = {
        "LCLid": ["A"] * 50 + ["B"] * 50,
        "datetime": pd.date_range("2023-01-01", periods=100, freq="H"),
        "energy(kWh/hh)": np.linspace(1, 100, 100),
        "temperature": np.linspace(10, 20, 100),
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "fake.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def make_fake_households():
    """
        Creates a dictionary of two fake households with 50 timesteps each and 2 features.

        why is it made:
        - Provides controlled data for rest of test.
        """
    households = {
        "A": np.column_stack([np.linspace(0, 10, 50), np.linspace(15, 25, 50)]),
        "B": np.column_stack([np.linspace(5, 15, 50), np.linspace(10, 20, 50)]),
    }
    return households


def test_load_and_process_csv(tmp_path):
    """
        Tests LoadAndProcessCSV.

        What it tests:
        - CSV can be read successfully.
        - Data is grouped by household ID.
        - Numeric columns are converted to NumPy arrays.
        - Each household array has the correct number of features (2 columns: energy, temperature).

        Why it matters:
        - Loading and preprocessing raw data is the first step in the pipeline.
        - If this fails, all downstream steps (normalization, sequencing, training) will fail.
        """
    csv_path = make_fake_csv(tmp_path)
    households = LoadAndProcessCSV(csv_path)
    assert isinstance(households, dict)
    assert all(isinstance(v, np.ndarray) for v in households.values())#loops over all in values in np array, and check if they are the correct datatype
    assert all(v.shape[1] == 2 for v in households.values()) #checks if each row has 2 collums


def test_make_scaler_and_normalize_data(tmp_path):
    """
        Tests makeScalerAndNormalizeData.

        What it tests:
        - All feature values are normalized to [0,1].
        - MinMaxScaler objects are saved to 'scalers.pkl'.
        - Saved scalers are of correct type (MinMaxScaler).

        Why it matters:
        - LSTM training requires normalized input for stability.
        - Saved scalers are needed to reverse normalization when we predict at the end.
        """
    households = make_fake_households()
    households_scaled, scalers = makeScalerAndNormalizeData(households)

    # Check normalization range
    for arr in households_scaled.values():
        assert np.all(arr >= 0) and np.all(arr <= 1) #check thatt all values are between 0 and 1 meaning they are normalized

    # Check scalers are saved
    assert os.path.exists("scalers.pkl")
    with open("scalers.pkl", "rb") as f:
        loaded = pickle.load(f)
    assert isinstance(loaded["energy(kWh/hh)"], MinMaxScaler)#check scaler is correct


def test_create_sequences_shapes():
    """
        Tests create_sequences.

        What it tests:
        - Returns X of shape (num_samples, timesteps, features) and y of shape (num_samples,).
        - Sequence slicing is correct for timesteps=24.

        Why it matters:
        - LSTMs require sequential inputs of fixed length.
        - Ensures temporal patterns are preserved and shapes are compatible with Keras.
        """
    data = np.random.rand(30, len(numeric_cols))  # 30 samples with 2 features each, they are between 0-1
    X, y = create_sequences(data)
    assert X.shape[0] == len(data) - 24 #it is slideingwindow. so 6 sequences is made
    assert X.shape[1] == 24 #check if 24 in one sequnce
    assert X.shape[2] == len(numeric_cols) #check 2 coloums
    assert y.shape[0] == len(data) - 24 #check if the result of a squence is == to the amount of squence


def test_stack_households_shapes():
    """
        Tests stackHouseholds.

        What it tests:
        - Multiple households are stacked into a single 3D input array X.
        - Target arrays y are concatenated into a single 1D array.(for now might change)
        - Feature dimension in X matches number of columns.

        Why it matters:
        - Ensures the model can train on multiple households at once.
        - Prevents shape mismatch errors during training.
        """
    households = make_fake_households()
    X, y = stackHouseholds(households)
    assert X.ndim == 3  # (samples, timesteps, features) to make sure it is right for the model
    assert y.ndim == 1 # might change to two
    assert X.shape[2] == len(numeric_cols) #makes sure the colums are 2


