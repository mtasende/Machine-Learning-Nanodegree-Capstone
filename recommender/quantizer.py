
import numpy as np

class Quantizer(object):
    """ Simple class to quantize floats. """

    def __init__(self, q_levels):
        self.q_levels = q_levels

    def quantize(self, real_value):
        """ Returns the number of interval in which the real value lies. """
        temp_list = self.q_levels + [real_value]
        temp_list.sort()
        sorted_index = temp_list.index(real_value)
        return sorted_index

    def get_quantized_value(self, real_value):
        """ Returns a quantized value, given the real value. """
        num_interval = self.quantize(real_value)
        l1 = self.q_levels[num_interval]
        l2 = self.q_levels[num_interval - 1]
        # Return the closest to the real value
        if np.abs(real_value - l1) < np.abs(real_value - l2):
            return l1
        else:
            return l2

    def interval_to_value(self, num_interval):
        """ Given an interval number, it calculates a 'quantized value'. """
        if num_interval == -1:
            return self.q_levels[0]
        if num_interval == len(self.q_levels):
            return self.q_levels[-1]
        return self.q_levels[num_interval]
