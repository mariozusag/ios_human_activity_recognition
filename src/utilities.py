# Mario Zusag mariozusag@gmail.com
# 06.03.19
# Purpose: Provide various utility functions
# Description:
import pandas as pd


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
