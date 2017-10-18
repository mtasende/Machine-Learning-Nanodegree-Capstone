import pandas as pd
import numpy as np


def leverage_reached(values, leverage_limit=3.0):
    if leverage_limit is None:
        return False
    else:
        leverage = np.sum(np.abs(values[:-1])) / (np.sum(values[:-1]) + values[-1])
        return leverage > leverage_limit


def simulate_orders(orders, data_df, initial_cap=1000, leverage_limit=None, from_csv=False):
    """
    This function simulates the earnings and losses in the market of a given set of orders.
    The orders can be given as a dataframe, or as a file path to a CSV file. The format is:
    [Date, Symbol, Order, Shares] where Order can be {BUY, SELL} and Shares is an integer amount.
    :param orders: Can be a DataFrame containing the orders or a path to a CSV file.
    :param data_df: The prices data from the market.
    :param initial_cap: The initial capital.
    :param leverage_limit: If the limit is reached no more stocks will be bought.
    :param from_csv: Set to true if you want to load the orders from a CSV file.
    :returns portval_df: The total portfolio value for the range of dates.
    :returns values_df: A matrix with the values allocated on each symbol or cash, for the entire
    period.
    """
    # Read the orders
    if from_csv:
        orders_df = pd.read_csv(orders, index_col='Date', parse_dates=True, na_values=['nan'])
    else:
        orders_df = orders

    # Let's order the orders by date, just in case
    orders_df.sort_index(inplace=True)

    # Let's get the initial and final dates:
    start_date = orders_df.index[0]
    end_date = orders_df.index[-1]

    # Get the symbols
    symbols = list(np.unique(orders_df['Symbol']))
    symbols = ['SPY'] + symbols  # There may be an extra column, but its value will be zero unless an order buys it

    # Get the data for those symbols and dates and add a "CASH" column
    prices_df = data_df.xs('Close', level='feature').loc[start_date:end_date, symbols]
    prices_df['CASH'] = 1.0

    # Let's build a DataFrame that contains the variations for each symbol ("CASH" included)
    variations = prices_df.copy()*0

    for date, order in orders_df.iterrows():
        # TODO: Check if there is a NaN, that may imply the stock wasn't traded that day
        if order['Order'] == 'BUY':
            variations.ix[date,order['Symbol']] += order['Shares']
            variations.ix[date,'CASH'] -= order['Shares']*prices_df.ix[date,order['Symbol']]
        if order['Order'] == 'SELL':
            variations.ix[date,order['Symbol']] -= order['Shares']
            variations.ix[date,'CASH'] += order['Shares']*prices_df.ix[date,order['Symbol']]

    # Now let's build the "values" dataframe (if leverage is >3 all orders for the day are cancelled)
    amounts = prices_df.copy()*0
    values = prices_df.copy()*0

    # Initialize the first day values
    amounts.iloc[0]['CASH'] = initial_cap
    values.iloc[0] = amounts.iloc[0] * prices_df.iloc[0]

    intended_amounts = amounts.iloc[0] + variations.iloc[0]
    intended_values = intended_amounts * prices_df.iloc[0]

    if not leverage_reached(intended_values.values, leverage_limit):
        amounts.iloc[0] = intended_amounts
        values.iloc[0] = intended_values

    # Now loop on the other days
    for i in range(1, len(values)):
        # Initialize the values as if there was no order this day
        amounts.iloc[i] = amounts.iloc[i-1]
        values.iloc[i] = amounts.iloc[i] * prices_df.iloc[i]

        # Now try to fill the possible orders, if leverage limit is not reached
        intended_amounts = amounts.iloc[i-1] + variations.iloc[i]
        intended_values = intended_amounts * prices_df.iloc[i]

        if not leverage_reached(intended_values.values, leverage_limit):
            amounts.iloc[i] = intended_amounts
            values.iloc[i] = intended_values

    # And the total portfolio value
    port_vals_df = pd.DataFrame(values.sum(axis=1), columns=['Value'])

    return port_vals_df, values

