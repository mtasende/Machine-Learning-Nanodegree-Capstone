""" This module defines an Random Forest regression predictor. """
from sklearn.ensemble import RandomForestRegressor
import pandas as pd


class RandomForestPredictor(object):
    """ It's a wrapper around the scikit-learn class that keeps
    the dates and ticker info."""

    def __init__(self):
        self.model = RandomForestRegressor()

    def fit(self, x, y):
        """
        It just fits the sklearn model.
        :param x: The training data. Each row is a sample.
        :param y: The training labels.
        """
        self.model.fit(x, y)

    def predict(self, x):
        """
        Predict using the sklearn model. Saves and returns the dates and symbols.
        :param x: The training data. Each row is a sample.
        :return: The predicted value. In this case, the mean.
        """
        return pd.DataFrame(self.model.predict(x), index=x.index, columns=['target'])

    def fit_predict(self, x, y):
        """ Returns the predicted values for x, after fitting x. In
        this case, just the mean of x."""
        self.fit(x, y)
        return self.predict(x)

    def get_params(self, deep=True):
        """ Wrapper """
        return self.model.get_params(deep)

    def set_params(self, **params):
        return self.model.set_params(**params)
