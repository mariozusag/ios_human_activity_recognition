# Mario Zusag mariozusag@gmail.com
# 11.03.19
# Purpose: Split the data into train and test sets for
# Description:

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import to_categorical
from src.preprocessing.data_loader import DataLoader


class_mapping = {"walking": 0, "jogging": 1} # , "downstairs": 2, "upstairs": 3, "standing": 4, "sit": 5}


def get_train_test_data(df: pd.DataFrame,
                        save_to: str,
                        means: dict,
                        stds: dict,
                        hertz=50,
                        milliseconds=10000,
                        n_classes=2,
                        test_split=0.25,
                        balance=True):
    """
    Transforms the data in df into arrays for training and testing

    Parameters
    ----------
    df: pandas DataFrame
    save_to: str
        Path to where the data should be saved to
    means: dict
        A dictionary with means values for each acceleration direction
    stds: dict
        A dictionary with standard deviation values for each acceleration direction
    hertz: int
        The frequency with which the data was collected
    milliseconds: int
        We want to get n milliseconds of data
    n_classes: int
        The number of classes we have
    test_split: float
        The ratio of train:test data
    balance: bool
        Whether or not the data should be balanced
    Returns
    -------
    tuple
        Returns X_train, X_test, y_train, y_test
    """
    every_n_ms = int(1000 / hertz)
    n_datapoints = int(milliseconds / every_n_ms)

    if balance:
        balanced_df = df.groupby('label')
        df = balanced_df.apply(lambda x: x.sample(balanced_df.size().min()).reset_index(drop=True))

    X = []
    y = []

    for i in range(0, len(df) - n_datapoints, every_n_ms):
        # 1: x1,   y1,   z1
        #    x2,   y2,   z2
        #    ...
        #    x500, y500, z500
        acc_x = df['acc.x'].values[i: i + n_datapoints]
        acc_y = df['acc.y'].values[i: i + n_datapoints]
        acc_z = df['acc.z'].values[i: i + n_datapoints]
        labels = [class_mapping[label] for label in df['label'][i: i + n_datapoints].values]
        label = np.argmax(np.bincount(labels))
        sample = [[acc_x[j], acc_y[j], acc_z[j]] for j in range(len(acc_x))]

        X.append(sample)
        y.append(label)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)

    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])  # flatten for coreml mlmultiarray
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])  # flatten for coreml mlmultiarray

    y_train = to_categorical(y_train, num_classes=n_classes)
    y_test = to_categorical(y_test, num_classes=n_classes)

    print("X_train shape: {}".format(X_train.shape))
    print("y_train shape: {}".format(y_train.shape))
    print("X_test shape:  {}".format(X_test.shape))
    print("y_test shape:  {}".format(X_test.shape))
    np.savez_compressed(save_to,
                        X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test,
                        means=means,
                        stds=stds)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    data_loader = DataLoader()
    motionsense_df = data_loader.load_motion_sense_data_as_df()
    X_train_split, X_test_split, y_train_split, y_test_split = get_train_test_data(motionsense_df,
                                                               means={},
                                                               stds={},
                                                               save_to='../../data/train_test/motion_sense_data.npz')
