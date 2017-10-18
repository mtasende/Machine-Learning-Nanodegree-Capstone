from recommender.environment import Environment
from recommender.indicator import Indicator
import recommender.indicator_functions as indf
import numpy as np
import pandas as pd
import sys
from time import time

INITIAL_CAP = 10000


def simulate_one_step(action, env, agent, actions):
    reward, new_state = env.get_consequences(actions[action])
    return agent.play(reward, new_state)


def create_indicators(data_df, n_levels=5):
    """
    Particular function to create a series of indicators.
    To remove one just comment it's line.
    """
    z_score_limits = (-2.0, 2.0)
    z_score_step = (z_score_limits[1] - z_score_limits[0]) / n_levels
    rsi_limits = (-2.0, 2.0)
    rsi_step = (rsi_limits[1] - rsi_limits[0]) / n_levels
    ovb_limits = (-2.0, 2.0)
    ovb_step = (ovb_limits[1] - ovb_limits[0]) / n_levels

    indicators = {}
    Z_SCORE = 'z_score'
    RSI = 'rsi'
    OVB = 'on_volume_balance'
    indicators[Z_SCORE] = Indicator(indf.z_score,
                                    indf.z_score_vec,
                                    q_levels=np.arange(z_score_limits[0],
                                                       z_score_limits[1],
                                                       z_score_step).tolist(),
                                    data_df=data_df,
                                    name=Z_SCORE)
    indicators[RSI] = Indicator(indf.rsi,
                                indf.rsi_vec,
                                q_levels=np.arange(rsi_limits[0],
                                                   rsi_limits[1],
                                                   rsi_step).tolist(),
                                data_df=data_df,
                                name=RSI)
    indicators[OVB] = Indicator(indf.on_volume_balance,
                                indf.on_volume_balance_vec,
                                q_levels=np.arange(ovb_limits[0],
                                                   ovb_limits[1],
                                                   ovb_step).tolist(),
                                data_df=data_df,
                                name=OVB)
    return indicators


def get_num_states(indicators):
    acum = 1
    for ind in indicators.values():
        acum *= len(ind.q_levels) + 1
    return acum


def initialize_env(total_data_df,
                   symbol,
                   starting_days_ahead=252,
                   possible_fractions=None,
                   n_levels=5):
    # Initialization
    data_df = total_data_df[symbol].unstack()
    indicators = create_indicators(data_df, n_levels)
    env = Environment(total_data_df,
                      indicators=indicators,
                      initial_cap=INITIAL_CAP,
                      symbol=symbol,
                      possible_fractions=possible_fractions)
    env.portfolio.set_current_date(data_df.index[starting_days_ahead])
    num_states = get_num_states(indicators)
    num_actions = len(env.actions_fractions.q_levels)  # All the possible fractions of total value
    return env, num_states, num_actions


def simulate_period(data_df,
                    symbol,
                    agent,
                    other_env=None,
                    verbose=False,
                    learn=True,
                    starting_days_ahead=252,
                    possible_fractions=None):
    """
    Simulate the market and one Agent for the entire period.
    data_df format is like below:
    """
    if other_env is None:
        env, num_states, num_actions = initialize_env(data_df,
                                                      symbol,
                                                      starting_days_ahead=starting_days_ahead,
                                                      possible_fractions=possible_fractions)
    else:
        env = other_env

    # Loop and play
    n_iters = data_df[symbol].unstack().shape[0] - starting_days_ahead
    end_date = data_df[symbol].unstack().index[-1]
    fraction_index = 0
    recorded_stock_value = {}
    recorded_cash_value = {}
    old_time = time()
    i = 0
    print('Starting simulation for agent: {}. {} days of simulation to go.'.format(agent, n_iters))
    # for i in range(N_iters):
    while env.portfolio.current_date < end_date:
        reward, new_state = env.get_consequences_from_fraction_index(fraction_index)

        if verbose:
            print('Date: {}, Value: {}'.format(env.portfolio.current_date, env.portfolio.get_total_value()))
            print('reward = {} \n\nnew_state = {} \n\naction = {} ({})'.format(reward,
                                                                               new_state,
                                                                               fraction_index,
                                                                               env.actions_fractions.interval_to_value(
                                                                                   fraction_index)))
            pos = env.portfolio.positions_df
            print(env.portfolio.get_positions())
            print(pos.loc[symbol, 'value'] / pos['value'].sum())
            print('-' * 70 + '\n\n')

        pos = env.portfolio.positions_df
        recorded_stock_value[env.portfolio.current_date] = pos.loc[symbol, 'value']
        recorded_cash_value[env.portfolio.current_date] = pos.loc['CASH', 'value']
        if learn:
            fraction_index = agent.play(reward, new_state)
        else:
            fraction_index = agent.play_learned_response(new_state)
        if i % 10 == 0:
            new_time = time()
            sys.stdout.write('\rDate {} (simulating until {}).  Time: {}s.  Value: {}.'.
                             format(env.portfolio.current_date,
                                    end_date,
                                    (new_time - old_time),
                                    env.portfolio.get_total_value()))
            old_time = new_time
        i += 1

    return pd.DataFrame({'stock_value': recorded_stock_value, 'cash': recorded_cash_value})
