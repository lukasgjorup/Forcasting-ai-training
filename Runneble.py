from src.initialization import data

data.init_data(
    dataset_dir="data/",
    dataset_path="halfhourly_dataset/halfhourly_dataset/",
    file_pattern="aShortBlock_{0}.csv",
    start_index=1,
    end_index=2,
    formatted_csv="formatted_data.csv",
    weather_dataset_path="weather_hourly_darksky.csv",
    data_cols=["LCLid", "tstp", "energy(kWh/hh)"],
    data_cols_out=["LCLid", "datetime", "energy(kWh/hh)"],
    force_rebuild_data=True,
        weather_data=["temperature"],
    data_points = 240

)