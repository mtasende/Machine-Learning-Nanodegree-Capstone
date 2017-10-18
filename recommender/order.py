import pandas as pd


class Order(pd.Series):
    """ Auxiliary class to make it easy to build an order. """
    # Accepted orders
    BUY = 'BUY'
    SELL = 'SELL'
    NOTHING = 'NOTHING'
    # Columns
    SYMBOL = 'symbol'
    ORDER = 'order'
    SHARES = 'shares'

    def __init__(self, data=None):
        super(Order, self).__init__(data)
        self.index = [self.SYMBOL, self.ORDER, self.SHARES]
