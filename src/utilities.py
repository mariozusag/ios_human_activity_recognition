# Mario Zusag mariozusag@gmail.com
# 06.03.19
# Purpose: Provide various utility functions
# Description:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def append_df(df_orig, df_to_append, label):
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


def clean_non_numeric_values(df, cols):
    for col in cols:
        df = df[(~pd.to_numeric(df[col], errors='coerce').isnull())]
        df[col] = pd.to_numeric(df[col])
    return df


def plot_classification_report(classificationReport,
                               title='Classification report',
                               cmap='YlGnBu',
                               save_as='../experimental_results/CNN_2019-02-19-13:44:04_best/report.pdf'):

    classificationReport = classificationReport.replace('\n\n', '\n')
    classificationReport = classificationReport.replace(' / ', '/')
    lines = classificationReport.split('\n')

    classes, plotMat, support, class_names = [], [], [], []
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

        plotMat.append(v)

    plotMat = np.array(plotMat)
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup)
                   for idx, sup in enumerate(support)]

    plt.figure()
    sns.heatmap(plotMat, fmt=".1f", square=True, annot=True, cmap=cmap,
                cbar=True, xticklabels=xticklabels, yticklabels=yticklabels)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.ylabel('Metrics')
    plt.xlabel('Classes')
    plt.tight_layout()
    plt.savefig(save_as)