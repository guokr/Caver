import numpy as np
from scipy import stats

from .classify import Caver


class Ensemble:
    """
    :param list models: each model should have the same label number

    For now, this only support soft voting methods.
    """
    def __init__(self, models):
        assert isinstance(models, list) and len(models) > 0
        label_num = models[0].config.label_num
        for m in models:
            assert isinstance(m, Caver)
            if label_num != m.config.label_num:
                raise ValueError('All the model must have the same numbers of labels.')

        self.models = models
        self.epsilon = 1e-8

        self.methods = {
            'avg': self.avg,
            'log': self.log,
            'hmean': self.hmean,
            'gmean': self.gmean,
        }

    def avg(self, preds):
        return np.average(preds, axis=0)

    def log(self, preds):
        return np.exp(np.log(self.epsilon + preds).mean(axis=0))

    def hmean(self, preds):
        return stats.hmean(self.epsilon + preds)

    def gmean(self, preds):
        return stats.gmean(self.epsilon + preds)

    def predict(self, text, method='log'):
        """
        :param str text: text
        :param str method: ['log', 'avg', 'hmean', 'gmean']
        """
        assert method in self.methods

        preds = np.array([m.predict(text) for m in self.models])
        return self.methods.get(method)(preds)

    def get_top_label(self, text, method='log', top=5):
        """
        :param str text: text
        :param str method: ['log', 'avg', 'hmean', 'gmean']
        :param int top: top-n most possible labels
        """
        preds = self.predict(text, method)
        top_index = np.argsort(preds)[::-1]
        result = []
        result.append([self.models[0].index2label[i] for i in top_index[:top]])
        result.append([preds[i] for i in top_index[:top]])
        return result
