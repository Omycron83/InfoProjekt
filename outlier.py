import numpy as np
from scipy.spatial.distance import pdist, squareform

class outlier_filtering:
    def __init__(self, Fraktilwert = 5) -> None:
        self.Grenzwert = 1.4
        self.Fraktilwert = Fraktilwert

    def filter(self, data):
        min, max = np.min(data, axis=0), np.max(data, axis=0)
        data_norm = (data - min) / (min - max)
        D_1 = pdist(data_norm)
        Z_1 = squareform(D_1)

        quantiles = np.percentile(Z_1, self.Fraktilwert, axis = 1)

        id_grenzwert = np.where(quantiles > self.Grenzwert)[0]

        return data[id_grenzwert, :], np.delete(data, id_grenzwert, axis = 0)