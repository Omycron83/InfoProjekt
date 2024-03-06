import numpy as np
from scipy.spatial.distance import pdist, squareform

class outlier_filtering:
    def __init__(self, Fraktilwert = 5, Grenzwert = 1.4) -> None:
        self.Grenzwert = Grenzwert
        self.Fraktilwert = Fraktilwert

    def filter(self, data):
        min, max = np.min(data, axis=0), np.max(data, axis=0)
        data_norm = (data - min) / (min - max)
        D_1 = pdist(data_norm)
        Z_1 = squareform(D_1)

        quantiles = np.percentile(Z_1, self.Fraktilwert, axis = 1)

        id_grenzwert = np.where(quantiles > self.Grenzwert)[0]

        return data[id_grenzwert, :], np.delete(data, id_grenzwert, axis = 0)
    
def unit_test():
    f, l = np.zeros(5, 3), np.zeros(5, 1)
    _class = DecisionTreeAutoClass(f, l)
    print(_class.pred(f), _class.cost(f, l))
    regr = DecisionTreeAutoRegr(f, l)
    print(regr.pred(f), _class.cost(f, l))

if __name__ == '__main__':
    unit_test()