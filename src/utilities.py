# Mario Zusag mariozusag@gmail.com
# 06.03.19
# Purpose: Provide various utility functions
# Description:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def append_df(df_orig: pd.DataFrame, df_to_append: pd.DataFrame, label: str):
    """
    Concatenates two DataFrames and adds the current label to each row of the df_to_append

    Parameters
    ----------
    df_orig: pandas.DataFrame
        The original DataFrame, to which we want to add rows
    df_to_append: pandas.DataFrame
        A new DataFrame object, which contains new rows from a csv file
    label: str
        A string containing the label for the new cases in df_to_append

    Returns
    -------
    pandas.DataFrame
        Returns a new DataFrame, which contains df_orig and df_to_append in a single DataFrame

    """
    df_to_append['label'] = label   # add additional label
    if df_orig is None:
        return df_to_append
    else:
        return pd.concat([df_orig, df_to_append], ignore_index=True, sort=False)


def clean_non_numeric_values(df: pd.DataFrame, cols: list):
    """
    Cleans away non-numerical values of a DataFrame

    Parameters
    ----------
    df: pandas DataFrame
        The pandas DataFrame, where we want to remove non-numerical values from
    cols: list
        The columns of df, where we want to remove non-numerical values form

    Returns
    -------
    pandas DataFrame
    The pandas DataFrame, with only numerical values in the specified columns

    """
    for col in cols:
        df = df[(~pd.to_numeric(df[col], errors='coerce').isnull())]
        df[col] = pd.to_numeric(df[col])
    return df


def plot_classification_report(classification_report,
                               title='Classification report',
                               cmap='YlGnBu',
                               save_as='../experimental_results/report.pdf'):
    """
    Plot a sklearn classification report

    Parameters
    ----------
    classification_report: str
        The classification report as it comes from sklearn
    title: str
        The title of the plot
    cmap: str or plt.cmap
        Colormap
    save_as: str
        The path, where we want to store the classification report plot

    Returns
    -------

    """

    classification_report = classification_report.replace('\n\n', '\n')
    classification_report = classification_report.replace(' / ', '/')
    lines = classification_report.split('\n')

    classes, plot_mat, support, class_names = [], [], [], []
    for line in lines[1:]:
        t = line.strip().split()
        if len(t) < 2:
            continue
        classes.append(t[0])
        if "avg" in t:
            v = [float(x)*100 for x in t[2: len(t) - 1]]
            class_names.append(t[0] + "-" + t[1])
        else:
            v = [float(x)*100 for x in t[1: len(t) - 1]]
            class_names.append(t[0])
        support.append(int(t[-1]))

        plot_mat.append(v)

    plot_mat = np.array(plot_mat)
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup)
                   for idx, sup in enumerate(support)]

    plt.figure()
    sns.heatmap(plot_mat, fmt=".1f", square=True, annot=True, cmap=cmap,
                cbar=True, xticklabels=xticklabels, yticklabels=yticklabels)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.ylabel('Metrics')
    plt.xlabel('Classes')
    plt.tight_layout()
    plt.savefig(save_as)


def min_max_normalize_df(df: pd.DataFrame):
    """
    Normalizes the acceleration values to range 0,1 and returns the maxima and minima of the complete df

    Parameters
    ----------
    df: pandas DataFrame
        The DataFrame we want to normalize

    Returns
    -------
    tuple
    Returns the normalized DataFrame, the maxima and the minima of the acceleration
    values (before normalization) for re-use at testing
    """
    accelerations = ['acc.x', 'acc.y', 'acc.z']
    minima = {}
    maxima = {}
    for acc in accelerations:
        minima[acc] = df[acc].min()
        df[acc] += abs(df[acc].min())
        maxima[acc] = df[acc].max()
        df[acc] /= abs(df[acc].max())

    return df, maxima, minima


def zero_mean_unit_variance_normalize_df(df: pd.DataFrame):
    """
    Normalizes the acceleration values to zero mean and unit variance, returns mean and std for each acceleration axis

    Parameters
    ----------
    df: pandas DataFrame
        The DataFrame we want to normalize

    Returns
    -------
    tuple
    Returns the normalized DataFrame, the means and the standard deviations of the acceleration
    values (before normalization) for re-use at testing
    """
    accelerations = ['acc.x', 'acc.y', 'acc.z']
    means = {}
    stds = {}
    for acc in accelerations:
        mu = np.mean(df[acc], axis=0)
        sigma = np.std(df[acc], axis=0)

        means[acc] = mu
        stds[acc] = sigma
        df[acc] = (df[acc] - mu)/sigma

    return df, means, stds
