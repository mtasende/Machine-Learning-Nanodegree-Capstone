""" This module takes care of getting the data, wherever it may be."""

from pandas_datareader import data
import datetime as dt
import pandas as pd
import numpy as np
from time import time
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARN)

DATA_SOURCE = 'google'
SPY_CREATION_DATE = dt.datetime(1993, 1, 22)
START_DATE = SPY_CREATION_DATE
END_DATE = dt.datetime(2017, 1, 1)
SP500_TICKER = 'SPY'
DATE_COL = 'date'
FEATURE_COL = 'feature'
SP500_LIST_PATH = '../data/sp500_list.csv'
DATES_BATCH_DAYS = 3650
PATH_TO_PICKLE = '../data/data_df.pkl'


def download_capstone_data():
    """ Gets the data for the capstone project and saves it in pickle format."""
    try:
        data_df = pd.read_pickle(PATH_TO_PICKLE)
        log.info('Previous data was found. Will update it.')
    except Exception as ex:
        log.info('No previous data was found. Downloading for the first time.')
        data_df = download_ticker(SP500_TICKER, START_DATE, END_DATE)
    sp500_list = get_sp500_list()
    for i, ticker in enumerate(sp500_list):
        if ticker not in data_df.columns:
            try:
                data_df = data_df.join(download_ticker(ticker, START_DATE, END_DATE), how='left')
                log.info('Saving data to disk. Do not interrupt now.')
                data_df.to_pickle(PATH_TO_PICKLE) # Save each symbol
                log.info('Added symbol: %s' % ticker)
            except Exception as ex:
                log.warning(ex)
                log.warning('It was not possible to download the symbol %s.' % ticker)
        else:
            log.info('Symbol %s was already in the dataframe. Not added again.' % ticker)
        log.info('%2.1f percent completed.' % (100.0*((i+1)/len(sp500_list))))
    log.debug(data_df.head(20))
    log.debug(data_df.tail())
    log.debug(data_df.shape)
    # For debugging
    # data_df.to_csv('../data/data_df.csv')


def download_ticker(symbol, start_date, end_date):
    """
    Downloads the data for one ticker
    :param symbol: The ticker symbol for the company or similar.
    :param start_date: a datetime object.
    :param end_date: a datetime object
    :return: A dataframe with the symbol's data (Open, High, Low, Close, Volume)
    in the defined inteval.
    """
    days_required = (end_date - start_date).days
    num_batches = (days_required // DATES_BATCH_DAYS) + 1

    sd = start_date
    ed = min(end_date, (start_date + dt.timedelta(days=DATES_BATCH_DAYS)))
    log.debug('sd = %s , ed = %s' % (sd, ed))
    raw_df = data.DataReader(name=symbol,
                             data_source=DATA_SOURCE,
                             start=sd,
                             end=ed)
    log.debug('batch 0 size: %i' % raw_df.shape[0])
    for batch_index in range(1, num_batches):
        new_ed = ed + dt.timedelta(days=DATES_BATCH_DAYS)
        sd = ed + dt.timedelta(days=1)
        ed = min(end_date, new_ed)
        log.debug('sd = %s , ed = %s' % (sd, ed))
        raw_df = raw_df.append(data.DataReader(name=symbol,
                                               data_source=DATA_SOURCE,
                                               start=sd,
                                               end=ed))
        log.debug('batch %i size: %i' % (batch_index, raw_df.shape[0]))
    log.debug(raw_df[raw_df.index.duplicated()])
    return raw_to_multiindex(raw_df, symbol)


def raw_to_multiindex(raw_df, name):
    """
    Takes a raw dataframe (vertically oriented) and transforms it in a multiindexed one.
    The indices are (date, feature).
    """
    iterables = [raw_df.index, raw_df.columns]
    index = pd.MultiIndex.from_product(iterables, names=[DATE_COL, FEATURE_COL])
    data_df = pd.DataFrame(index=index)
    data_df[name] = np.nan
    for date in raw_df.index:
        for col in raw_df.columns:
            data_df.loc[date, col][name] = raw_df.loc[date, col].copy()
    return data_df


def get_sp500_list():
    """ Returns a list with the tickers of the constituents of the S&P500 index
    (fixed to the date of creation; there is no dynamic update, for now)."""
    return pd.read_csv(SP500_LIST_PATH, header=0)['ticker'].tolist()
    # For debugging only
    # return ['SPY', 'GOOG', 'AAPL', 'NVDA']

if __name__ == '__main__':
    print('The list to downlad is this')
    print(get_sp500_list())
    print((END_DATE-START_DATE).days)
    tic = time()
    download_capstone_data()
    toc = time()
    print('Elapsed time: %i seconds' % (toc - tic))
