""" This module contains some functions to evaluate the stock prices predictor. """
import sys
import numpy as np
from pandas.errors import UnsortedIndexError
from sklearn.metrics import r2_score
import pandas as pd
import predictor.feature_extraction as fe
import datetime as dt


def mre(y_true, y_pred):
    """
    MRE metrics function. The values of assets should never be zero so, as zero labels cause problems,
    they are not considered.
    """
    y_true_filtered = y_true[y_true != 0]
    y_pred_filtered = y_pred[y_true != 0]
    return np.mean(np.abs((y_pred_filtered - y_true_filtered) / y_true_filtered))


def get_metrics(y_true_df, y_pred_df):
    """
    Calculates the MRE and R^2 score, on a per-symbol basis. It receives matrices of results,
    in which the rows represent time and the columns represent symbols.
    :param y_true_df: The labels for each symbol at each moment in time.
    :param y_pred_df: The predicted labels for each symbol at each moment in time.
    :returns r2_scores: Numpy array with the R^2 score for each symbol
    :returns mre_scores: Numpy array with the MRE score for each symbol
    :returns tickers: Array that contains the ticker symbols.
    """
    tickers = y_true_df.index.levels[1]
    r2_scores = []
    mre_scores = []

    for ticker in tickers:
        try:
            y_true = y_true_df.loc[(slice(None), ticker), :]
            y_pred = y_pred_df.loc[(slice(None), ticker), :]
        except UnsortedIndexError:
            y_true = y_true_df.sort_index().loc[(slice(None), ticker), :]
            y_pred = y_pred_df.sort_index().loc[(slice(None), ticker), :]
        r2_scores.append(r2_score(y_true, y_pred))
        mre_scores.append(mre(y_true.values, y_pred.values))

    return np.array(r2_scores), np.array(mre_scores), tickers


def get_metrics_df(y_true_df, y_pred_df):
    """
    Wrapper around get_metrics that returns dataframes instead of Numpy arrays.
    """
    r2_scores, mre_scores, tickers = get_metrics(y_true_df, y_pred_df)
    return pd.DataFrame(np.array([r2_scores, mre_scores]).T, index=tickers, columns=['r2', 'mre'])


def get_metrics_in_time(y_true_df, y_pred_df, shift):
    """
    Calculates the MRE and R^2 score, on a per-time basis. It receives matrices of results,
    in which the rows represent time and the columns represent symbols.
    :param y_true_df: The labels for each symbol at each moment in time.
    :param y_pred_df: The predicted labels for each symbol at each moment in time.
    :return: The mean MRE and R^2 score for each time point, and an array of the corresponding dates.
    """
    dates = y_true_df.index.get_level_values(0).unique()
    r2_scores = []
    mre_scores = []

    for date in dates:
        try:
            y_true = y_true_df.loc[(date, slice(None)), :]
            y_pred = y_pred_df.loc[(date, slice(None)), :]
        except UnsortedIndexError:
            y_true = y_true_df.sort_index().loc[(date, slice(None)), :]
            y_pred = y_pred_df.sort_index().loc[(date, slice(None)), :]
        r2_scores.append(r2_score(y_true, y_pred))
        mre_scores.append(mre(y_true.values, y_pred.values))

    return np.array(r2_scores), np.array(mre_scores), dates + dt.timedelta(shift)


def reshape_by_symbol(y):
    """ Deprecated helper function. Was not used in the final implementation."""
    grouped_df = y.reset_index() \
        .groupby('level_0') \
        .apply(lambda x: x.reset_index(drop=True)) \
        .drop('level_0', axis=1)
    grouped_df.index = grouped_df.index.droplevel(level=1)
    grouped_df.rename(columns={'level_1': 'ticker'}, inplace=True)
    reshaped_df = grouped_df.set_index('ticker', append=True).unstack()
    reshaped_df.columns = reshaped_df.columns.droplevel(level=0)
    reshaped_df.index.name = 'date'
    return reshaped_df


def run_single_val(x, y, ahead_days, estimator):
    """
    Runs a single training and validation.
    :param x: A dataframe of samples. The columns represent the base days.
    The rows always contain the dates of the initial day in the base period. Additionally,
    the dataframe may be multiindexed with information about from which symbol each sample comes from.
    The symbol information is not used for the training, but may be useful to get some insigths in
    the validation process.
    :param y: The labels of each sample. It corresponds to the (standarized) value of a ticker, some days ahead.
    :param ahead_days: Number of days ahead that the labels are from the last base day.
    :param estimator: A predictor object for the labels. It follows the scikit-learn interface, but keeps the dataframe
    information.
    :returns y_train_true_df: Labels for the training set. Rows contain dates, columns contain symbols.
    :returns y_train_pred_df: Predictions for the training set. Rows contain dates, columns contain symbols.
    :returns y_val_true_df: Labels for the validation set. Rows contain dates, columns contain symbols.
    :returns y_val_pred_df: Predictions for the validation set. Rows contain dates, columns contain symbols.
    """
    multiindex = x.index.nlevels > 1

    x_y = pd.concat([x, y], axis=1)
    x_y_sorted = x_y.sort_index()
    if multiindex:
        x_y_train = x_y_sorted.loc[:fe.add_market_days(x_y_sorted.index.levels[0][-1], -ahead_days)]
        x_y_val = x_y_sorted.loc[x_y_sorted.index.levels[0][-1]:]
    else:
        x_y_train = x_y_sorted.loc[:fe.add_market_days(x_y_sorted.index[-1], -ahead_days)]
        x_y_val = x_y_sorted.loc[x_y_sorted.index[-1]:]

    x_train = x_y_train.iloc[:, :-1]
    x_val = x_y_val.iloc[:, :-1]
    y_train_true = x_y_train.iloc[:, -1]
    y_val_true = x_y_val.iloc[:, -1]

    estimator.fit(x_train, y_train_true)
    y_train_pred = estimator.predict(x_train)
    y_val_pred = estimator.predict(x_val)

    y_train_true_df = pd.DataFrame(y_train_true)
    y_train_pred_df = pd.DataFrame(y_train_pred)
    y_val_true_df = pd.DataFrame(y_val_true)
    y_val_pred_df = pd.DataFrame(y_val_pred)

    # Just to make it look prettier
    y_train_pred_df.columns = y_train_true_df.columns
    y_val_pred_df.columns = y_val_true_df.columns

    return y_train_true_df, \
        y_train_pred_df, \
        y_val_true_df, \
        y_val_pred_df


def roll_evaluate(x, y, train_days, step_eval_days, ahead_days, predictor, verbose=False):
    """
    Warning: The final date of the period should be no larger than the final date of the SPY_DF
    This function applies run_single_val many times, in a rolling evaluation fashion.

    :param x: A dataframe of samples. Normally it will span for a period larger than the training period.
    The columns represent the base days.
    The rows always contain the dates of the initial day in the base period. Additionally,
    the dataframe may be multiindexed with information about from which symbol each sample comes from.
    The symbol information is not used for the training, but may be useful to get some insigths in
    the validation process.
    :param y: The labels of each sample. It corresponds to the (standarized) value of a ticker, some days ahead.
    :param train_days: The amount of training days for each train-validation run.
    :param step_eval_days: The amount of days to move the training and validation sets on each cycle.
    :param ahead_days: Number of days ahead that the labels are from the last base day.
    :param predictor: A predictor object for the labels. It follows the scikit-learn interface, but keeps the dataframe
    information.
    :param verbose: If true it shows some messages and progress reports.
    :returns r2_train_metrics: A numpy array with the mean and standard deviation of the R^2 metrics for each date of
    evaluation. The mean and std are taken on the symbols dimension.
    :returns mre_train_metrics: A numpy array with the mean and standard deviation of the MRE metrics for each date of
    evaluation. The mean and std are taken on the symbols dimension.
    :returns y_val_true_df: Labels for the validation set. Rows contain dates, columns contain symbols.
    :returns y_val_pred_df: Predictions for the validation set. Rows contain dates, columns contain symbols.
    :returns mean_dates: The mean date of the training period. It is useful to plot the training metrics in time.
    """

    # calculate start and end date
    # sort by date
    x_y_sorted = pd.concat([x, y], axis=1).sort_index()
    start_date = x_y_sorted.index.levels[0][0]
    end_date = fe.add_market_days(start_date, train_days)
    final_date = x_y_sorted.index.levels[0][-1]

    # loop: run_single_val(x,y, ahead_days, estimator)
    mean_dates = []
    r2_train_means = []
    r2_train_stds = []
    mre_train_means = []
    mre_train_stds = []
    y_val_true_df = pd.DataFrame()
    y_val_pred_df = pd.DataFrame()
    num_training_sets = (252 / 365) * (
    x.index.levels[0].max() - fe.add_market_days(x.index.levels[0].min(), train_days)).days // step_eval_days
    set_index = 0
    if verbose:
        print('Evaluating approximately %i training/evaluation pairs' % num_training_sets)

    while end_date < final_date:
        x_temp = x_y_sorted.loc[start_date:end_date].iloc[:, :-1]
        y_temp = x_y_sorted.loc[start_date:end_date].iloc[:, -1]
        x_temp.index = x_temp.index.remove_unused_levels()
        y_temp.index = y_temp.index.remove_unused_levels()
        y_train_true, y_train_pred, y_val_true, y_val_pred = run_single_val(x_temp, y_temp, ahead_days, predictor)

        # Register the mean date of the period for later use
        mean_dates.append(start_date + ((end_date - start_date) / 2))

        # Calculate R^2 and MRE for training and append
        r2_scores, mre_scores, tickers = get_metrics(y_train_true, y_train_pred)
        r2_train_means.append(np.mean(r2_scores))
        r2_train_stds.append(np.std(r2_scores))
        mre_train_means.append(np.mean(mre_scores))
        mre_train_stds.append(np.std(mre_scores))

        # Append validation results
        y_val_true_df = y_val_true_df.append(y_val_true)
        y_val_pred_df = y_val_pred_df.append(y_val_pred)

        # Update the dates
        start_date = fe.add_market_days(start_date, step_eval_days)
        end_date = fe.add_market_days(end_date, step_eval_days)

        set_index += 1
        if verbose:
            sys.stdout.write('\rApproximately %2.1f percent complete.    ' % (100.0 * set_index / num_training_sets))
            sys.stdout.flush()

    return np.array([r2_train_means, r2_train_stds]).T, \
        np.array([mre_train_means, mre_train_stds]).T, \
        y_val_true_df, \
        y_val_pred_df, \
        np.array(mean_dates)
