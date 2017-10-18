import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CLOSE_COL_NAME = 'Close'


def compute_portfolio_stats(portfolioValue, rfr=0.0, sf=252.0):

    cumRet = (portfolioValue.iloc[-1]/portfolioValue.iloc[0]) - 1

    dailyRets = get_daily(portfolioValue)

    # Statistics below don't take into account the first day value
    averageDReturn = dailyRets[1:].mean()
    sdDReturn = dailyRets[1:].std()

    # Sharpe ratio calculations
    if rfr==0.0:
        daily_rfr = 0.0
    else:
        daily_rfr = ((1.0+rfr)**(1.0/sf)) -1
    sharpeRatio = np.sqrt(sf)*(dailyRets[1:] - daily_rfr).mean()/(dailyRets[1:]-daily_rfr).std()

    return cumRet, averageDReturn, sdDReturn, sharpeRatio


def assess_portfolio(start_date,
                     end_date,
                     symbols,
                     allocations,
                     initial_capital=1000,
                     risk_free_rate=0.0,
                     sampling_frequency=252.0,
                     data=None,
                     gen_plot=False,
                     verbose=False):
    """
    This function returns some statistics about a portfolio.
    :param start_date: Starting Date; Type = dt.datetime
    :param end_date: Ending Date; Type = dt.datetime
    :param symbols: A list of ticker symbols; Type: list of strings
    :param allocations: A list with the fraction of allocations to each symbol; Type: list of float
    :param initial_capital: Starting Value of the portfolio; Type: float/int
    :param risk_free_rate: Free rate of interest; Type: float
    :param sampling_frequency: Sampling frequency per year (252.0 = daily)
    :param data: A dataframe with the data of the s&p500 stocks
    :param gen_plot: if True create a plot
    :param verbose: if True, print the output data
    :returns cumulative_ret: Cumulative Return; Type: float
    :returns average_daily_ret: Average Daily Return. If 'sf' is different from 252,
    then it is the average in the sampling period instead of "daily"; Type: float
    :returns sd_daily_ret: Standard deviation of daily return; Type: float
    :returns sharpe_ratio: Sharpe Ratio; Type: float
    :returns end_value: End value of portfolio; Type: float/int
    """

    adj_close = data.xs(CLOSE_COL_NAME, level='feature').loc[start_date:end_date,symbols]
    adj_close /= adj_close.iloc[0]  # Normalize to the first day
    norm_value = adj_close.dot(allocations)  # Get the normalized total value
    portfolio_value = pd.DataFrame(norm_value * initial_capital)

    # Compute statistics from the total portfolio value
    cumulative_ret, \
        average_daily_ret, \
        sd_daily_ret, \
        sharpe_ratio = compute_portfolio_stats(portfolio_value, risk_free_rate, sampling_frequency)
    end_value = portfolio_value.iloc[-1]

    if gen_plot:
        adj_close_SPY = data.xs(CLOSE_COL_NAME, level='feature').loc[start_date:end_date,'SPY']
        adj_close_SPY /= adj_close_SPY.iloc[0]
        ax = adj_close_SPY.plot(color='g', label='SPY')
        ax.plot(norm_value, color='b', label='Portfolio')
        plt.title('Daily portfolio value and SPY')
        plt.xlabel('Date')
        plt.ylabel('Normalized price')
        plt.legend(loc='upper left')
        plt.show()

    if verbose:
        print('Start Date: ' + str(start_date))
        print('End Date: ' + str(end_date))
        print('Symbols: ' + str(symbols))
        print('Allocations ' + str(allocations))
        print('Sharpe Ratio: %.15f' % sharpe_ratio)
        print('Volatility (stdev of daily returns): %.15f' % sd_daily_ret)
        print('Average Daily Return: %.15f' % average_daily_ret)
        print('Cumulative Return: %.15f' % cumulative_ret)

    return float(cumulative_ret), float(average_daily_ret), float(sd_daily_ret), float(sharpe_ratio), float(end_value)


def value_eval(value_df,
               risk_free_rate=0.0,
               sampling_frequency=252.0,
               verbose=False,
               graph=False,
               data_df=None):
    """ This function takes a value of portfolio series, returns some statistics, and shows some plots"""

    cumulative_ret = (value_df.iloc[-1]/value_df.iloc[0]) - 1
    daily_rets = get_daily(value_df)

    # Statistics below don't take into account the first day value
    average_daily_ret = daily_rets[1:].mean()
    sd_daily_ret = daily_rets[1:].std()

    # Sharpe ratio calculations
    if risk_free_rate == 0.0:
        daily_rfr = 0.0
    else:
        daily_rfr = ((1.0 + risk_free_rate)**(1.0/sampling_frequency)) -1
    sharpe_ratio = np.sqrt(sampling_frequency)*(daily_rets[1:] - daily_rfr).mean()/(daily_rets[1:]-daily_rfr).std()

    if verbose:
        print('sharpeRatio = %f' % sharpe_ratio)
        print('cumRet = %f' % cumulative_ret)
        print('sdDReturn = %f' % sd_daily_ret)
        print('averageDReturn = %f' % average_daily_ret)
        print('Final Value: %f' % value_df.iloc[-1])

    if graph:
        if data_df is not None:
            value_df = value_df \
                .join(data_df.xs('Close', level='feature').loc[:, 'SPY'], how='left')
        value_df = value_df / value_df.iloc[0]
        value_df.plot()

    return sharpe_ratio.values[0], \
        cumulative_ret.values[0], \
        average_daily_ret.values[0], \
        sd_daily_ret.values[0], \
        value_df.iloc[-1].values[0]


# Returns the daily change of a variable
def get_daily(data):
    daily = data.copy()
    daily[1:] = (data[1:] / data[:-1].values)-1
    daily.ix[0,:] = np.zeros(len(data.columns))

    return daily


# Printing some basic data about a ticker's attribute data
def basic_data(ticker,attr_data):
    print('Ticker name: '+ticker)
    print('    Mean: %f'% attr_data[ticker].mean())
    print('    Std: %f'% attr_data[ticker].std())
    print('    Kurtosis: %f'% attr_data[ticker].kurtosis())
    print('')

