import datetime
import os
import time

import numpy as np
import pandas as pd


def read_training_data(
    directory_path="some/directory/path/",
    file_pattern="some_file{0}.csv",
    start_index=0,
    end_index=112,
):
    """Reads the training data

    Parameters
    ----------
    dataset_dir : str
        The directory of the dataset
    file_pattern : str
        The format of the files
    start_index : int
        The index of the first file
    end_index : int
        The index of the last file


    Returns
    -------
    pandas.DataFrame
        a pandas dataframe of the files
    """
    csv_dataframe = pd.DataFrame(
        pd.read_csv(os.path.join(directory_path, file_pattern.format(0)))
    )
    for i in range(start_index + 1, end_index):
        data = pd.read_csv(
            pd.read_csv(os.path.join(directory_path, file_pattern.format(i)))
        )
        dataframe = pd.DataFrame(data)
        csv_dataframe = pd.concat([csv_dataframe, dataframe], axis=0)
    return csv_dataframe


def append_weather_data(
    dataframe,
    weather_csv,
    data_cols,
    data_cols_out,
    datetime_col,
    weather_data,
    weather_time_col,
    data_points,
    id_col,
):
    """Appends weather data to training data and removes undesirable columns

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The dataframe that needs to have weather data appended
    weather_csv : str
        name of the weather dataset
    data_cols : str array
        array of names of columns to keep in the simplified dataframe
    data_cols_out : str array
        array of column names in the new dataframe (must have same length as data_cols)
    date_time_col : str
        Name of the column that contains the datetime information
    weather_data : str array
        array of column names to use from weather data
    weather_data_col : str
        name of the column that contains the time data from the weather dataset
    data_points : int
        Nubmer of datapoints for identical IDs specified in ID_col
    id_col : str
        Name of the column that contains IDs. Defaults to first element of data_cols

    Returns
    -------
    pandas.DataFrame
        a pandas dataframe with the columns specified in data_cols_out and weather_data
    """
    start = time.time()
    abridged_df = pd.DataFrame()
    abridged_df[data_cols_out] = dataframe[data_cols]
    current_id = ""
    data_point_cnt = 0
    i = 0
    idx = 0
    if id_col == "":
        current_id = abridged_df.loc[0, data_cols_out[0]]
    else:
        current_id = abridged_df.loc[0, id_col]
    if os.path.exists(weather_csv):
        weather_df = pd.read_csv(weather_csv)
    while i < len(abridged_df):
        idx += 1
        print(
            f"Progress: {100 * (idx / len(dataframe)):.2f}% time elapsed: "
            + str(datetime.timedelta(seconds=time.time() - start))
            + " estimated time left: "
            + str(
                datetime.timedelta(
                    seconds=(((time.time() - start) / (idx + 0.1)) * (len(dataframe) - idx))
                )
            ),
            end="\r",
        )
        if data_points != -1:
            if id_col == "":
                if current_id != abridged_df.loc[i, data_cols_out[0]]:
                    current_id = abridged_df.loc[i, data_cols_out[0]]
                    data_point_cnt = 0
            else:
                if current_id != abridged_df.loc[i, id_col]:
                    current_id = abridged_df.loc[i, id_col]
                    data_point_cnt = 0

            if data_point_cnt >= data_points:
                abridged_df.drop(index=abridged_df.iloc[i].name, inplace=True)
                abridged_df.reset_index(drop=True, inplace=True)
                continue

        abridged_df[datetime_col] = pd.to_datetime(abridged_df[datetime_col])
        if os.path.exists(weather_csv):
            match = weather_df.loc[
                weather_df[weather_time_col].str.contains(
                    str(abridged_df.loc[i, datetime_col])
                ),
                weather_data,
            ]
            if not match.empty:
                abridged_df.loc[i, weather_data] = match.iloc[0].values
            else:
                match = weather_df.loc[
                    weather_df[weather_time_col].str.contains(
                        str(
                            abridged_df.loc[i, datetime_col]
                            + datetime.timedelta(minutes=30)
                        )
                    ),
                    weather_data,
                ]
                if not match.empty:
                    abridged_df.loc[i, weather_data] = match.iloc[0].values
                else:
                    for i in range(0, len(weather_data)):
                        abridged_df.loc[i, weather_data[i]] = np.mean(
                            weather_df.loc[weather_data[i]]
                        )  # for some reason there's a couple missing days in the weather data
        i = i + 1
        data_point_cnt = data_point_cnt + 1
    return abridged_df


def init_data(
    dataset_dir="data/",
    dataset_path="halfhourly_dataset/halfhourly_dataset/",
    file_pattern="block_{0}.csv",
    start_index=0,
    end_index=112,
    formatted_csv="formatted_data.csv",
    weather_dataset_path="weather_hourly_darksky.csv",
    force_rebuild_data=False,
    data_cols=["LCLid", "tstp", "energy(kWh/hh)"],
    data_cols_out=["LCLid", "datetime", "energy(kWh/hh)"],
    datetime_col="datetime",
    weather_data=["temperature"],
    weather_time_col="time",
    data_points=-1,
    id_col="",
):
    """Initalizes the dataset

    Parameters
    ----------
    dataset_dir : str
        The directory of the dataset
    file_pattern : str
        The pattern of the files Note: that only the first element will be replaced i.e. file{0}.csv is valid but file{0}_{1}.csv will throw an exception
    start_index : int
        The index of the first file
    end_index : int
        The index of the last file
    formatted_csv : str
        path to where the preprocessed data should be stored for future initializations
    force_rebuild_data : bool
        boolean flag that forces the rebuild of preprocessed data
    data_cols : str array
        array of names of columns to keep in the simplified dataframe
    data_cols_out : str array
        array of column names in the new dataframe (must have same length as data_cols)
    date_time_col : str
        Name of the column that contains the datetime information
    weather_data : str array
        array of column names to use from weather data
    weather_data_col : str
        name of the column that contains the time data from the weather dataset
    data_points : int
        Nubmer of datapoints for identical IDs specified in ID_col
    id_Col : str
        Name of the column that contains IDs. Defaults to first element of data_cols


    Returns
    -------
    numpy.array
        a numpy array with the columns specified in data_cols_out and weather_data
    """
    start = time.time()
    if force_rebuild_data or not os.path.exists(formatted_csv):
        if force_rebuild_data is False:
            print("initalizing first time data preprocessing")
        df = read_training_data(
            directory_path=os.path.join(dataset_dir, dataset_path),
            file_pattern=file_pattern,
            start_index=start_index,
            end_index=end_index,
        )
        df = append_weather_data(
            df,
            os.path.join(dataset_dir, weather_dataset_path),
            data_cols,
            data_cols_out,
            datetime_col,
            weather_data,
            weather_time_col,
            data_points,
            id_col,
        )
        df.to_csv(formatted_csv, index=False)
    else:
        print("initializing from data from " + formatted_csv)
        df = pd.read_csv(formatted_csv)
        print(df)
        end = time.time()
        init_time = end - start
        print("initalization time : " + str(datetime.timedelta(seconds=init_time)))

    return df.to_numpy()
