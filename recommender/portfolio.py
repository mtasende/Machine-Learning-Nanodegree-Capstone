""" Portfolio simulator. """
import numpy as np
import pandas as pd
from recommender.order import Order

SHARES = 'shares'
VALUE = 'value'
CASH = 'CASH'


def leverage(positions_df):
    """ This function assumes that if your net worth is negative the leverage is infinite. """
    values = positions_df[VALUE]
    return np.sum(np.abs(values[CASH])) / (np.max([0.0, np.sum(values)]))


class Portfolio(object):
    """
    data_df format:

		                    SPY 	    MMM 	    ABT
    date 	    feature
    1993-01-29 	Open 	    0.00 	    0.00 	    0.00
                High 	    43.97 	    24.62 	    6.88
                Low 	    43.75 	    24.47 	    6.75
                Close 	    43.94 	    24.50 	    6.88
                Volume 	    1003200.00 	1242800.00 	4638400.00
    """
    def __init__(self, data_df, initial_cap=1000, leverage_limit=None):
        # Fixed data
        self.data_df = data_df
        self.close_df = data_df.xs('Close', level='feature')
        self.market_days = self.close_df.sort_index().index.unique().tolist()
        self.symbols = data_df.columns.get_level_values(0).tolist()
        self.symbols.append(CASH)
        self.leverage_limit = leverage_limit

        # Variable data
        self.current_date = data_df.index.get_level_values(0)[0]
        self.positions_df = pd.DataFrame(index=self.symbols, columns=[SHARES, VALUE])
        self.positions_df[SHARES] = np.zeros(self.positions_df.shape[0])
        self.positions_df.loc[CASH, SHARES] = initial_cap
        self.update_values()

    def update_positions_values(self, positions_df):
        """ Updates the positions_df values, using the current prices and the shares,
        for a custom positions_df. Returns the updated postions_df. """
        prices = self.close_df.loc[self.current_date]
        prices[CASH] = 1.0
        result_df = positions_df.copy()
        result_df[VALUE] = result_df[SHARES] * prices
        return result_df.fillna(0.0)

    def update_values(self):
        """ Updates the positions_df values, using the current prices and the shares. """
        self.positions_df = self.update_positions_values(self.positions_df)

    def execute_order(self, order, add_one_day=False):
        """
        Executes an order on the current date, if the leverage is below the limit.
        :param order: It is a pandas Series with columns = [symbol, order, shares].
         The 'order' is either BUY, SELL or NOTHING.
        :param add_one_day: If it is true, the current date is updated to the next available date in data_df.
        :return: Boolean that is set to True if the order was executed, and False otherwise.
        """
        new_positions_df = self.positions_df.copy()
        if order[Order.ORDER] == Order.BUY:
            new_positions_df.loc[order[Order.SYMBOL], SHARES] += order[Order.SHARES]
            new_positions_df.loc[CASH, SHARES] -= order[Order.SHARES] * \
                self.close_df.loc[self.current_date, order[Order.SYMBOL]]
        if order[Order.ORDER] == Order.SELL:
            new_positions_df.loc[order[Order.SYMBOL], SHARES] -= order[Order.SHARES]
            new_positions_df.loc[CASH, SHARES] += order[Order.SHARES] * \
                self.close_df.loc[self.current_date, order[Order.SYMBOL]]
        new_positions_df = self.update_positions_values(new_positions_df)
        if np.isnan(self.close_df.loc[self.current_date, order[Order.SYMBOL]]):
            return False
        if self.leverage_reached(new_positions_df, self.leverage_limit):
            return False
        self.positions_df = new_positions_df
        if add_one_day:
            self.add_market_days(1)
        return True

    def add_market_days(self, delta):
        """ Updates the current date by moving it 'delta' market days. 'delta' can be negative. """
        base_index = self.market_days.index(self.current_date)
        if base_index + delta >= len(self.market_days):
            self.current_date = self.market_days[-1]
        elif base_index + delta < 0:
            self.current_date = self.market_days[0]
        else:
            self.current_date = self.market_days[int(base_index + delta)]
        self.update_values()

    @staticmethod
    def leverage_reached(positions_df, leverage_limit):
        """ Checks if the specified leverage limit was reached. """
        if leverage_limit is None:
            return False
        else:
            return not leverage(positions_df) < leverage_limit

    def my_leverage_reached(self):
        self.update_values()
        return self.leverage_reached(self.positions_df, self.leverage_limit)

    def get_leverage(self):
        self.update_values()
        return leverage(self.positions_df)

    def set_current_date(self, new_date):
        self.current_date = new_date

    def get_positions(self):
        self.update_values()
        return self.positions_df[self.positions_df[SHARES] != 0]

    def get_total_value(self):
        self.update_values()
        return self.positions_df[VALUE].sum()

    def clone_with_new_data(self, new_data_df):
        return_portfolio = Portfolio(new_data_df,
                                     initial_cap=1000,
                                     leverage_limit=self.leverage_limit)
        return_portfolio.data_df = new_data_df
        return_portfolio.positions_df = self.positions_df.copy()
        return_portfolio.set_current_date(self.current_date)
        return_portfolio.update_values()
        return return_portfolio
