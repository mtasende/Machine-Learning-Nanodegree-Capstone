from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import utils.preprocessing as pp


class Indicator(object):
    """
    A class that represents a feature 'of importance' to the agent.
    data_df example:

    """

    def __init__(self, extractor, extractor_vec, q_levels, data_df, window=14, name='unknown'):
        self.extractor = extractor
        self.extractor_vec = extractor_vec
        self.window = window
        self.q_levels = q_levels
        self.scaler = StandardScaler()
        self.fit(data_df)
        self.name = name

    def fit(self, data_df):
        extracted_data = self.extractor_vec(data_df, self.window)
        self.scaler.fit(extracted_data.fillna(0).values.reshape(-1, 1))

    def interval_to_value_vec(self, num_interval_vec):
        q_dict = {}
        for index in num_interval_vec.index:
            q_dict[index] = self.interval_to_value(int(num_interval_vec.loc[index].values[0]))
        return pd.DataFrame.from_dict(q_dict, orient='index', dtype=np.float64)

    def interval_to_value(self, num_interval):
        """ Given an interval number it calculates a 'quantized value'. """
        if num_interval == 0:
            return self.q_levels[0]
        if num_interval == len(self.q_levels):
            return self.q_levels[-1]
        return (self.q_levels[num_interval] + self.q_levels[num_interval-1]) / 2

    def quantize_vec(self, real_values_df):
        q_dict = {}
        for index in real_values_df.index:
            q_dict[index] = self.quantize(real_values_df.loc[index])
        return pd.DataFrame.from_dict(q_dict, orient='index', dtype=np.float64)

    def quantize(self, real_value):
        """ Returns the number of interval in which the real value lies. """
        temp_list = self.q_levels + [real_value]
        temp_list.sort()
        sorted_index = temp_list.index(real_value)
        return sorted_index

    def get_quantized_value(self, real_value):
        """ Returns a quantized value, given the real value. """
        return self.interval_to_value(self.quantize(real_value))

    def extract(self, data_df):
        """ Returns the indicator value in the last date of data_df"""
        raw_res = np.array([[self.extractor(pp.fill_missing(data_df), self.window)]])
        if np.isnan(raw_res[0,0]):
            print(self.name)
            print(raw_res)
            print(data_df.iloc[-self.window:])
        scaled_res = self.scaler.transform(raw_res)
        return self.quantize(scaled_res[0, 0])

    def extract_vec(self, data_df):
        """ Return a pandas Series with the values of the indicator for all the valid dates in data_df."""
        temp_df = data_df.copy()
        temp_df['ind'] = self.scaler.transform(
            pp.fill_missing(
                self.extractor_vec(pp.fill_missing(data_df), self.window)).values.reshape(-1, 1))
        return self.quantize_vec(temp_df['ind'])
