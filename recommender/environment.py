""" This module has classes and functions that are used as the environment for the trading agent. """
import itertools as it
import numpy as np
from recommender.portfolio import Portfolio
import recommender.portfolio as port
from recommender.order import Order
from recommender.quantizer import Quantizer


def reward_value_change(old_pos_df, new_pos_df):
    return new_pos_df[port.VALUE].sum() - old_pos_df[port.VALUE].sum()


def reward_cash_change(old_pos_df, new_pos_df):
    return new_pos_df.loc[port.CASH, port.VALUE] - old_pos_df.loc[port.CASH, port.VALUE]


class Environment(object):

    def __init__(self,
                 data_df,
                 indicators=None,
                 initial_cap=1000,
                 leverage_limit=None,
                 reward_fun=None,
                 symbol='AAPL',
                 possible_fractions=None):
        if indicators:
            self.indicators = indicators
        else:
            self.indicators = {}
        self.initial_capital = initial_cap
        self.leverage_limit = leverage_limit
        self.state_vectors = {}
        self.states = {}
        self.initialize_states()
        self.data_df = data_df
        self.portfolio = Portfolio(self.data_df, self.initial_capital, self.leverage_limit)
        if reward_fun:
            self.reward_fun = reward_fun
        else:
            self.reward_fun = reward_value_change
        self.symbol = symbol
        if possible_fractions:
            self.possible_fractions = possible_fractions
        else:
            self.possible_fractions = np.arange(0.0, 1.1, 0.1).round(decimals=3).tolist()
        self.actions_fractions = Quantizer(self.possible_fractions)

    def add_indicator(self, name, indicator):
        """ Adds an indicator. NOTE: this changes the whole state encoding. """
        self.indicators[name] = indicator
        self.initialize_states()

    def remove_indicator(self, name):
        """ Removes an indicator. NOTE: this changes the whole state encoding. """
        del self.indicators[name]
        self.initialize_states()

    def vector_to_state(self, state_vector):
        """ Given a vector state, returns an integer state. """
        return self.states[state_vector]

    def state_to_vector(self, state):
        """ Given an integer state, returns a vector state (index of quantized intervals vector). """
        return self.state_vectors[state]

    def extract_indicators(self, data_df):
        """ Returns a vector state with the quantized index of all the indicators. """
        return tuple(map(lambda x: x.extract(data_df[self.symbol].unstack()), self.indicators.values()))

    def initialize_states(self):
        """ It initializes the structures necessary to store the states in their integer and vector forms. """
        states_list = list(it.product(*map(lambda x: np.arange(len(x.q_levels)+1), self.indicators.values())))
        self.state_vectors = dict(enumerate(states_list))
        self.states = dict(zip(self.state_vectors.values(), self.state_vectors.keys()))

    def get_consequences(self, action):
        """
        Given an action it returns the reward and the new state.
        :param action: It is a list of orders for the day.
        """
        old_positions_df = self.portfolio.get_positions()
        for order in action:
            self.portfolio.execute_order(order)
        self.portfolio.add_market_days(1)
        new_positions_df = self.portfolio.get_positions()
        reward = self.reward_fun(old_positions_df, new_positions_df)
        new_state = self.vector_to_state(self.extract_indicators(self.data_df[:self.portfolio.current_date]))
        return reward, new_state

    def get_state(self):
        return self.vector_to_state(self.extract_indicators(self.data_df))

    def act_to_target(self, target_fraction):
        """
        Perform the necessary actions to get a position close to the target_fraction of the portfolio total value, in
        the current symbol's shares. The rounding is always towards zero, and the possible fractions are determined by
        the actions_fractions Quantizer.
        :param target_fraction: The target of shares to get, as a fraction of the total value.
        :return: reward and new state.
        """
        current_price = self.portfolio.close_df.loc[self.portfolio.current_date][self.symbol]
        wanted_shares = np.fix(self.portfolio.get_total_value() *
                               self.actions_fractions.get_quantized_value(target_fraction) / current_price)
        previous_shares = self.portfolio.positions_df.loc[self.symbol, port.SHARES]
        shares_increase = wanted_shares - previous_shares
        action = [Order([self.symbol, Order.BUY, shares_increase])]
        return self.get_consequences(action)

    def get_consequences_from_fraction_index(self, fraction_index):
        target_fraction = self.actions_fractions.interval_to_value(fraction_index)
        return self.act_to_target(target_fraction)

    def reward_final_value(self, old_pos_df, new_pos_df):
        if self.portfolio.current_date() == self.data_df.index[-1]:
            return new_pos_df[port.VALUE].sum() / self.initial_capital
        else:
            return 0.0

    def reset(self, starting_days_ahead):
        """ It sets the environment to its initial state."""
        self.portfolio = Portfolio(self.data_df, self.initial_capital, self.leverage_limit)
        u_data_df = self.data_df[self.symbol].unstack()
        self.portfolio.set_current_date(u_data_df.index[starting_days_ahead])

    def set_test_data(self, data_df, starting_days_ahead):
        self.data_df = data_df
        self.reset(starting_days_ahead)

    def clone_with_new_data(self, new_data_df):
        return_env = Environment(self.data_df,
                                 indicators=self.indicators,
                                 initial_cap=self.initial_capital,
                                 symbol=self.symbol,
                                 possible_fractions=self.possible_fractions)
        return_env.data_df = new_data_df
        return_env.portfolio = self.portfolio.clone_with_new_data(new_data_df)
        return return_env
