import pandas as pd
from src.initialization import data


def test_initialize_training_data():
    proc_data = data.init_data(
        dataset_dir="data/test",
        dataset_path="halfhourly_dataset/",
        file_pattern="shortened_block_{0}.csv",
        start_index=0,
        end_index=1,
        formatted_csv="formatted_data.csv",
        weather_dataset_path="weather_hourly_darksky.csv",
        force_rebuild_data=True,
    )
    verified_data = pd.read_csv("data/test/formatted_data.csv")
    verified_data["datetime"] = pd.to_datetime(verified_data["datetime"])
    assert (verified_data.to_numpy() == proc_data).all()
