# Mario Zusag mariozusag@gmail.com
# 06.03.19
# Purpose:
# Description:

import pandas as pd
import os
from utilities import append_df, clean_non_numeric_values


DEFAULT_WISDM_PATH = '../../data/wisdm_lab/WISDM_ar_v1.1_raw.txt'
DEFAULT_MOTION_SENSE_PATH = '../../data/motion_sense/'
DEFAULT_RUN_WALK_PATH = '../../data/run_walk/'
columns = ['timestamp', 'label', 'acc.x', 'acc.y', 'acc.z']  # columns to keep
numerical_cols = ['acc.x', 'acc.y', 'acc.z']

class DataLoader:

    def __init__(self):
        """
        Initializes a DataLoader object, which encompasses different functions for loading datasets
        """
        pass

    @staticmethod
    def load_wisdm_data_as_df(data_path=DEFAULT_WISDM_PATH):
        """
        Loads and transforms WISDM data to a DataFrame with the specified global columns.
        WISDM data is in the format:
            subject-ID, label, time-stamp, acceleration.x, acceleration.y, acceleration.z

        Parameters
        ----------
        data_path: str
            Path to sensor data

        Returns
        -------
        data: pandas DataFrame
            Returns a DataFrame with the specified global columns
        """

        original_columns = ["subject-id", 'label', 'timestamp', 'acc.x', 'acc.y', 'acc.z']  # original columns in data
        print('Loading data from {}'.format(data_path))

        data = None

        if data_path.endswith('.txt'):
            data_as_dict = {}
            with open(data_path) as f:
                for idx, line in enumerate(f):
                    line = line.rstrip("\n")
                    line = line.rstrip(";")
                    values = line.split(',')
                    if len(values) == len(original_columns):
                        data_as_dict[idx] = {original_columns[i]: values[i] for i in range(len(values))}
            data = pd.DataFrame.from_dict(data_as_dict, orient='index')
            data = data[columns]
            data = data.replace({"label": {label:label.lower() for label in pd.unique(data['label'])}})
            data = clean_non_numeric_values(data, numerical_cols)

        print("WISDM dataset contains {} samples and {} as features".format(len(data), pd.unique(data['label'])))

        return data

    @staticmethod
    def load_run_walk_data_as_df(data_path=DEFAULT_RUN_WALK_PATH):
        """
        Loads and transforms run/walk data to a DataFrame with the specified global columns.
        Run/Walk data is in the format:
            date, time, username, wrist, activity,
            acceleration_x, acceleration_y, acceleration_z,
            gyro_x, gyro_y, gyro_z

        Parameters
        ----------
        data_path: str
            Path to sensor data

        Returns
        -------
        data: pandas DataFrame
            Returns a DataFrame with the specified global columns
        """

        print('Loading data from {}'.format(data_path))

        if data_path.endswith('dataset.csv'):
            data = pd.read_csv(data_path)
            # map the binary labels to string labels
            data["label"] = data["activity"].map({0: "walking", 1: "jogging"})
            # combine date and time to timestamp
            data['timestamp'] = data['date'] + "-" + data['time']
            data = data.rename(columns={"acceleration_x": "acc.x", "acceleration_y": "acc.y", "acceleration_z": "acc.z"})
            data = data[columns]
        else:
            dirs = "/".join(elem for elem in data_path.split('/')[:-1])
            raise ValueError("{} does not look like the correct file from the run/walk dataset. Please make sure you "
                             "have the correct file, i.e. 'dataset.csv' in {}".format(data_path, dirs))

        print("Run/walk dataset contains {} samples and {} "
              "as features".format(len(data), pd.unique(data['label'])))
        data = clean_non_numeric_values(data, numerical_cols)
        return data

    @staticmethod
    def load_motion_sense_data_as_df(data_path=DEFAULT_MOTION_SENSE_PATH, save_as="dataset.csv"):
        """
        Transforms multiple csv files from the motionSense dataset to a single csv/xlsx file
        Loops through the path, where motionSense csv files are stored.
        The motionSense dataset is saved under multiple folders, where each folder is associated to 1 of 6
        classes, i.e. dws=downstairs, jog=jogging, sit=sitting, std=standing, ups=upstairs, wlk=walking
        Here, I provide code, which traverses into the folders, loads each of the csv files of the folders and
        appends all data to a single pandas DataFrame. I drop several columns and only keep the ones for
        acceleration data in x,y,z direction. I also append a label column, which contains the first 3 letters
        of each folder (which is their labelling convention)

        Parameters
        ----------
        data_path: str
            Path to folders from motionSense
        save_as: str
            How to save the processed motionSense data

        Returns
        -------
        data: pandas DataFrame
            Returns a DataFrame with the specified global columns
        """
        hertz = 50.0
        timestep = 1000.0/hertz  # every 20 milliseconds a measurement is done, if data was recorded with 50Hz

        if not os.path.exists(os.path.join(data_path, save_as)):
            print('Loading data from {}'.format(data_path))
            data = None
            for dirs, _, files in os.walk(data_path):
                for file in files:
                    if file.endswith('.csv'):
                        label = dirs.split('/')[-1][:3]
                        c_file_path = os.path.join(dirs, file)
                        df = pd.read_csv(c_file_path)
                        data = append_df(data, df_to_append=df, label=label)
            print("DataFrame before renaming:")
            print(data.head(10))
            data["timestamp"] = [i*timestep for i in range(len(data))]
            data = data.rename(columns={"userAcceleration.x": "acc.x",
                                        "userAcceleration.y": "acc.y",
                                        "userAcceleration.z": "acc.z", })
            data = data[columns]
            data = data.replace({"label": {'dws': "downstairs",
                                           'sit': "sitting",
                                           'ups': "upstairs",
                                           'wlk': "walking",
                                           'std': "standing",
                                           'jog': "jogging"}})
            data.to_csv(os.path.join(data_path, save_as))


        else:
            print('Loading data from {}'.format(os.path.join(data_path, save_as)))
            data = pd.read_csv(os.path.join(data_path, save_as))

        print("MotionSense dataset contains {} samples and {} "
              "as features".format(len(data), pd.unique(data['label'])))

        data = clean_non_numeric_values(data, numerical_cols)

        print("Saved data under {}".format(os.path.join(data_path, save_as)))
        return data


if __name__ == '__main__':
    data_loader = DataLoader()
    wsdm_dataframe = data_loader.load_wisdm_data_as_df()
    print(wsdm_dataframe)
