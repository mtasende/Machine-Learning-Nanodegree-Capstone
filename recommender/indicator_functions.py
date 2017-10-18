"""
This module has a collection of functions that implement indicators.
data_df is like below:

    feature 	Close 	High 	Low 	Open 	Volume
    date
    1993-01-29 	2.70 	2.75 	2.68 	0.0 	39424000.0
    1993-02-01 	2.73 	2.75 	2.67 	0.0 	42854400.0
                            ...
"""
# TODO: It would be nice to implement a multi-column rolling function, so that there is no need to implement a separate
# "_vec" version for each indicator.

import numpy as np


def rsi(data_df, window=14):
    """
    Relative Strength Index.
    :param data_df: Unstacked for one symbol only.
    :param window: An integer number of days to look back.
    :return: The value of the RSI indicator for the last day of the period.
    """
    if (window + 1) > data_df.shape[0]:
        print('window too short')
        return np.nan
    data_sub_df = data_df.iloc[-window-1:]
    close = data_sub_df['Close']
    delta = close.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    mean_down = down[1:].abs().mean()
    if mean_down == 0:
        return 1.0
    rs = up[1:].mean() / mean_down
    try:
        return 1.0 - (1.0 / (1.0 + rs))  # Normalized to 1.0 instead of 100.0
    except ZeroDivisionError:
        print('zero denominator.')
        return np.nan


def rsi_vec(data_df, window=14):
    """
    Calculates the RSI indicator for an entire dataframe.
    :param data_df: Unstacked for one symbol only.
    :param window: An integer number of days to look back.
    :return: A pandas Series with the values of RSI for all the valid dates.
    """
    close = data_df['Close']
    delta = close.diff()
    # delta = delta.iloc[1:]
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = up.rolling(window=window, center=False).mean()
    roll_down = down.abs().rolling(window=window, center=False).mean()
    rs = roll_up / roll_down
    return 1.0 - (1.0 / (1.0 + rs))


def z_score(data_df, window=14):
    """
    Standardized deviation of the prices in a window.
    :param data_df: Unstacked for one symbol only.
    :param window: An integer number of days to look back.
    :return: The value of the z-score indicator for the last day of the period.
    """
    if window > data_df.shape[0]:
        print('window too small')
        return np.nan
    data_sub_df = data_df.iloc[-window:]
    close = data_sub_df['Close']
    if close.std() == 0:
        print(close)
    return (close[-1] - close.mean()) / close.std()


def z_score_vec(data_df, window=14):
    """
    Standardized deviation of the prices in a window.
    :param data_df: Unstacked for one symbol only.
    :param window: An integer number of days to look back.
    :return: A pandas Series with the values of z-score for all the valid dates.
    """
    close = data_df['Close']
    return close.rolling(window=window, center=False).apply(lambda x: (x[-1] - x.mean()) / x.std(ddof=1))


def on_volume_balance(data_df, window=14):
    """
    On Volume Balance (OBV).
    :param data_df: Unstacked for one symbol only.
    :param window: An integer number of days to look back.
    :return: The value of the OBV indicator for the last day of the period.
    """
    if window+1 > data_df.shape[0]:
        return np.nan
    data_sub_df = data_df.iloc[-window-1:]
    obv_delta = data_sub_df['Volume'] * np.sign(data_sub_df['Close'].diff())
    return obv_delta[-window:].sum()


def on_volume_balance_vec(data_df, window=14):
    """
    On Volume Balance (OBV).
    :param data_df: Unstacked for one symbol only.
    :param window: An integer number of days to look back.
    :return: A pandas Series with the values of OBV for all the valid dates.
    """
    obv_delta = data_df['Volume'] * np.sign(data_df['Close'].diff())
    return obv_delta.rolling(window=window, center=False).sum()
