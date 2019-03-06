# Mario Zusag mariozusag@gmail.com
# 06.03.19
# Purpose:
# Description:

import json
import argparse
import pandas as pd


class DataLoader:

    def __init__(self):
        """
        Initializes a DataLoader object, which encompasses different functions for loading datasets
        """
        pass

    @staticmethod
    def load_data_as_dict(data_path, columns=None):
        """

        Parameters
        ----------
        data_path: str
            Path to sensor data
        columns: list
            List of strings for naming the columns in the data

        Returns: dict
            Returns a dictionary of dictionaries referenced by an assigned index,
             where dict[idx] = {column1:value1, ...}
        -------

        """
        print('Loading data from {}'.format(data_path))
        data = {}
        if columns is None:
            columns = ['test_subject', 'label', 'timestamp', 'x', 'y', 'z']

        if data_path.endswith('.txt'):
            with open(data_path) as f:
                for idx, line in enumerate(f):
                    line = line.rstrip("\n")
                    values = line.split(',')
                    if len(values) == len(columns):
                        data[idx] = {columns[i]: values[i] for i in range(len(values))}

        return data

    @staticmethod
    def get_data_size(data_path):
        return len(open(data_path).readlines())

    def load_data_as_df(self, data_path, columns=None):
        if columns is None:
            columns = ['test_subject', 'label', 'timestamp', 'x', 'y', 'z']
        data_as_dict = self.load_data_as_dict(data_path, columns=columns)
        return pd.DataFrame.from_dict(data_as_dict, orient='index')


if __name__ == '__main__':
    pre_processing = DataLoader()
    dataframe = pre_processing.load_data_as_df('../../data/wisdm_lab/WISDM_ar_v1.1_raw.txt')
    print(dataframe)
