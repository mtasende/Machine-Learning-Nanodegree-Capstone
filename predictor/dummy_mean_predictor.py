""" This module defines a dummy predictor. """
import pandas as pd


class DummyPredictor(object):
    """ Dummy predictor. Returns the mean of the input data."""

    def fit(self, x, y):
        """
        Dummy fit of the training data. It does nothing.
        :param x: The training data. Each row is a sample.
        """
        pass

    def predict(self, x):
        """
        Dummy predict, returns the mean.
        :param x: The training data. Each row is a sample.
        :return: The predicted value. In this case, the mean.
        """
        return x.mean(axis=1)

    def fit_predict(self, x, y):
        """ Returns the predicted values for x, after fitting x. In
        this case, just the mean of x."""
        self.fit(x, y)
        return self.predict(x)
