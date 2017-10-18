""" Module with the preprocessing functions and classes"""
import math

FEATURE_OF_INTEREST = 'Close'


def fill_missing(values_df):
    return_df = values_df.fillna(method='ffill')
    return_df = return_df.fillna(method='bfill')
    return return_df


def normalize(values_df):
    return values_df / values_df.iloc[0]


def drop_irrelevant_symbols(data_df, good_data_ratio):
    """
    Drops the symbols that have less than the good_data_ratio of non-missing data,
    or that has the target value missing.
    """
    if data_df.columns.nlevels == 1:
        return data_df.dropna(thresh=math.ceil(good_data_ratio * data_df.shape[0]), axis=1)
    else:
        filtered_data_df = data_df[FEATURE_OF_INTEREST].dropna(
            thresh=math.ceil(good_data_ratio * data_df[FEATURE_OF_INTEREST].shape[0]), axis=1)
        return data_df.loc[:, (slice(None), filtered_data_df.columns.tolist())]


def drop_irrelevant_samples(x_y_df, good_data_ratio):
    """
    Drops the samples that have less than the good_data_ratio of non-missing data,
    or that has the target value missing.
    To drop all the samples that have any missing data, use "good_data_ratio=-1"
    """
    x_df = x_y_df.iloc[:, :-1]
    y_df = x_y_df.iloc[:, -1]
    if good_data_ratio < 0:
        x_df = x_df.dropna(how='any', axis=0)
    else:
        x_df = x_df.dropna(thresh=math.ceil(good_data_ratio*x_df.shape[1]), axis=0)
    y_df = y_df.dropna()
    return x_df.join(y_df, how='inner')
