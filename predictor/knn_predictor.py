""" This module defines a knn regression predictor. """
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd


class KNNPredictor(object):
    """ It's a wrapper around the scikit-learn class that keeps
    the dates and ticker info."""

    def __init__(self):
        self.model = KNeighborsRegressor()

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
