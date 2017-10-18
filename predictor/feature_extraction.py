""" This module takes care of generating the features and labels for
the trainer """
import numpy as np
import pandas as pd

TARGET_FEATURE = 'Close'
SPY_DF = pd.read_pickle('/'.join(__file__.split('/')[:-2]) + '/data/spy_df.pkl')
VOLUME_FEATURE = 'Volume'


def generate_train_intervals(data_df, train_days, base_days, step_days, ahead_days, today,
                             blob_fun, target_feature=TARGET_FEATURE):
    """
    Divides a training period into many base intervals and targets. I try to keep the convention
    that variables measured in "market days" have the "days" suffix, while those that represent
    real time dates don't.

    :param data_df: the original data, assumed to have dates as indexes.
    :param train_days: the days to look before today. It is an amount of market days. If it is
    set to -1, the entire data_df is used.
    :param base_days: the length, in time, of each training sample. It is measured in "market days".
    :param step_days: the distance between the starting time of each sample. It is measured in
    "market days".
    :param ahead_days: the number of days between the last training day and the target day.
    Measured in "market days".
    :param today: the date in which we live... the last day to predict. A real date.
    :param blob_fun: a function that, given all the features, from all the companies in a base
    time period, returns samples, somehow.
    :returns X: a dataframe with rows of length base_time, to be used as samples. Each column is
    the value in a different day (a feature). Possibly, there may be more columns than days,
    if more than one feature per day is selected.
    :returns y: the value of the target feature some days ahead of the base period, for each base
    period.
    """
    end_of_training_date = add_index_days(today, -ahead_days, data_df)
    if train_days == -1:
        start_date = data_df.index[0]
    else:
        start_date = add_index_days(end_of_training_date, -train_days, data_df)
    start_target_date = add_index_days(start_date, base_days + ahead_days - 1, data_df)

    data_train_df = data_df[start_date:end_of_training_date]
    data_target_df = data_df.loc[start_target_date: today, target_feature]  # Look here!

    date_base_ini = start_date
    date_base_end = add_index_days(date_base_ini, base_days - 1, data_df)
    date_target = add_index_days(date_base_end, ahead_days, data_df)
    feat_tgt_df = pd.DataFrame()

    while date_base_end < end_of_training_date:
        sample_blob = (data_train_df[date_base_ini: date_base_end],
                       pd.DataFrame(data_target_df.loc[date_target]))
        feat_tgt_blob = blob_fun(sample_blob)
        feat_tgt_df = feat_tgt_df.append(feat_tgt_blob)

        date_base_ini = add_index_days(date_base_ini, step_days, data_df)
        date_base_end = add_index_days(date_base_ini, base_days - 1, data_df)
        date_target = add_index_days(date_base_end, ahead_days, data_df)
        # print('Start: %s,  End:%s' % (date_base_ini, date_base_end))

    feat_tgt_df = feat_tgt_df.sample(frac=1)

    X_df = feat_tgt_df.iloc[:, :-1]
    y_df = pd.DataFrame(feat_tgt_df.iloc[:, -1]).rename(columns={7: 'target'})

    return X_df, y_df


def add_index_days(base, delta, data_df):
    """
    base is in real time.
    delta is in market days.
    It adds delta days that are present in data_df.index. It can handle duplicates.
    """
    if data_df.index.nlevels == 1:
        market_days = data_df.sort_index().index.unique()
    else:
        market_days = data_df.sort_index().index.levels[0].unique()
    if base not in market_days:
        raise Exception('The base date is not in the market days list.')
    base_index = market_days.tolist().index(base)
    if base_index + delta >= len(market_days):
        return market_days[-1]
    if base_index + delta < 0:
        return market_days[0]
    return market_days[int(base_index + delta)]


def add_market_days(base, delta):
    return add_index_days(base, delta, SPY_DF)


def feature_close_one_to_one(sample_blob):
    """
    Takes all the symbols as if they were the same, and assigns as target the Close value for
    each one of them.
    :param sample_blob: A tuple with 2 dataframes: one with the features, the other with the
    targets.
    :return: A combined dataframe with one sample per row, and some columns. The last column is
    the target, and the previous ones are for features. Only considers the "Close" value as a
    feature.
    """
    target = sample_blob[1].T
    feat_close = sample_blob[0][TARGET_FEATURE]
    x_y_samples = feat_close.append(target)
    x_y_samples /= x_y_samples.iloc[0]
    x_y_samples.index = pd.MultiIndex.from_product([[x_y_samples.index[0]],
                                                    np.arange(x_y_samples.shape[0])])
    x_y_samples_shuffled = x_y_samples.unstack().stack(0).sample(frac=1)
    return x_y_samples_shuffled


def feature_volume_one_to_one(sample_blob):
    """
    Takes all the symbols as if they were the same, and assigns as target the Volume value for
    each one of them.
    :param sample_blob: A tuple with 2 dataframes: one with the features, the other with the
    targets.
    :return: A combined dataframe with one sample per row, and some columns. The last column is
    the target, and the previous ones are for features. Only considers the "Close" value as a
    feature.
    """
    target = sample_blob[1].T
    feat_close = sample_blob[0][VOLUME_FEATURE]
    x_y_samples = feat_close.append(target)
    x_y_samples /= x_y_samples.iloc[0]
    x_y_samples.index = pd.MultiIndex.from_product([[x_y_samples.index[0]],
                                                    np.arange(x_y_samples.shape[0])])
    x_y_samples_shuffled = x_y_samples.unstack().stack(0).sample(frac=1)
    return x_y_samples_shuffled


def get_file_dir():
    return '/'.join(__file__.split('/')[:-1])

def get_data_dir():
    return '/'.join(__file__.split('/')[:-2]) + '/data'


if __name__ == '__main__':
    print(get_file_dir())
    print(get_data_dir())
