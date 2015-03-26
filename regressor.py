import numpy as np


class Regressor(object):
    def score(self, X, y):
        SSres = y - self.predict(X)
        SSres = np.dot(SSres.T, SSres)
        SStot = y - np.mean(y)
        SStot = np.dot(SStot.T, SStot)
        return 1 - SSres/SStot