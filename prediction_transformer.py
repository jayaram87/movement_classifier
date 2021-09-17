import numpy as np
import pickle
from logger import Logger

class Predictor_Data_Transformer:

    def __init__(self, avg_rss12, var_rss12, avg_rss13, var_rss13, avg_rss23, var_rss23):
        try:
            self.avg_rss12 = avg_rss12
            self.var_rss12 = var_rss12
            self.avg_rss13 = avg_rss13
            self.var_rss13 = var_rss13
            self.avg_rss23 = avg_rss23
            self.var_rss23 = var_rss23
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Couldnt initialize the transformer \n {str(e)}')

    def data(self):
        try:
            se = pickle.load(open("std_scaler.sav", 'rb'))
            data = np.array([self.avg_rss12, self.var_rss12, self.avg_rss13, self.var_rss13, self.avg_rss23, self.var_rss23]).reshape(1, -1)
            Logger('test.log').logger('ERROR', f'{data.shape}')
            data = se.transform(data)
            return data
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Couldnt transform the prediction data \n {str(e)}')
