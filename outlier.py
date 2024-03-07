import numpy as np
from scipy.spatial.distance import pdist, squareform

class outlier_filtering:
    def __init__(self, Fraktilwert = 5) -> None:
        self.Fraktilwert = Fraktilwert

    def filter(self, data):
        min, max = np.min(data, axis=0), np.max(data, axis=0)
        data_norm = (data - min) / (max - min)
        D_1 = pdist(data_norm)
        Z_1 = squareform(D_1)

        quantiles = np.percentile(Z_1, self.Fraktilwert, axis = 1)
        self.Grenzwert = np.percentile(quantiles, 100 - self.Fraktilwert)
        id_grenzwert = np.where(quantiles > self.Grenzwert)[0]

        return data[id_grenzwert, :], np.delete(data, id_grenzwert, axis = 0)  #Outliers and outlier-free dataset
    
def unit_test():
    z = np.random.normal(size=(40, 40))
    g = outlier_filtering()
    x, y = g.filter(z)
    print(x.shape, y.shape)
if __name__ == '__main__':
    unit_test()