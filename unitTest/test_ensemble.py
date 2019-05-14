import unittest
import caver.ensemble_m.utils as utils

import torch.nn as nn
import torch
import argparse
import unitTest.utils as utils
import caver.ensemble_m.ensemble as ensemble


class TestEnsemble(unittest.TestCase):


    def test_ensemble_mean(self):
        """
        unittest arithmetic mean method
        :return:
        """
        test_preds = [
            [torch.Tensor([[0.5, 0.3, 0.2]]), torch.Tensor([[0.4, 0.3, 0.3]])],
            [torch.Tensor([[0.5, 0.3, 0.2]]), torch.Tensor([[0.4, 0.3, 0.3]])],
        ]
        test_model_ratios = [[], [0.6, 0.4]]
        test_gold = [
            torch.Tensor([[0.45, 0.3, 0.25]]),
            torch.Tensor([[0.46, 0.3, 0.24]]),
        ]

        models = utils.load_models()
        for idx in range(len(test_preds)):
            pred = test_preds[idx]
            ratio = test_model_ratios[idx]
            gold = test_gold[idx]
            ensem_model = ensemble.EnsembleModel(models, ratio)
            ensem_pred = ensem_model.mean(pred)
            for x, y in zip(ensem_pred.data.tolist(), gold.data.tolist()):
                for m, n in zip(x, y):
                    self.assertEqual(utils.myRound(m), utils.myRound(n))
        print("test arithmetic mean method")

    def test_ensemble_gmean(self):
        """
        unittest geometric mean method
        :return:
        """
        test_preds = [
            [torch.Tensor([[0.5, 0.3, 0.2]]), torch.Tensor([[0.4, 0.3, 0.3]])],
            [torch.Tensor([[0.5, 0.3, 0.2]]), torch.Tensor([[0.4, 0.3, 0.3]])],
        ]
        test_model_ratios = [[], [0.6, 0.4]]
        test_gold = [
            torch.Tensor([[0.4472, 0.3, 0.2449]]),
            torch.Tensor([[0.4573, 0.3, 0.2352]]),
        ]

        models = utils.load_models()
        for idx in range(len(test_preds)):
            pred = test_preds[idx]
            ratio = test_model_ratios[idx]
            gold = test_gold[idx]
            ensem_model = ensemble.EnsembleModel(models, ratio)
            ensem_pred = ensem_model.gmean(pred)
            for x, y in zip(ensem_pred.data.tolist(), gold.data.tolist()):
                for m, n in zip(x, y):
                    self.assertEqual(utils.myRound(m), utils.myRound(n))
        print("test geometric mean method")


    def test_ensemble_hmean(self):
        '''
        unittest harmonic mean mean method
        :return:
        '''
        test_preds = [
            [torch.Tensor([[0.5, 0.3, 0.2]]), torch.Tensor([[0.4, 0.3, 0.3]])],
            [torch.Tensor([[0.5, 0.3, 0.2]]), torch.Tensor([[0.4, 0.3, 0.3]])],
        ]
        test_model_ratios = [[], [0.6, 0.4]]
        test_gold = [
            torch.Tensor([[0.4444, 0.3, 0.24]]),
            torch.Tensor([[0.4545, 0.3, 0.2308]]),
        ]

        models = utils.load_models()
        for idx in range(len(test_preds)):
            pred = test_preds[idx]
            ratio = test_model_ratios[idx]
            gold = test_gold[idx]
            ensem_model = ensemble.EnsembleModel(models, ratio)
            ensem_pred = ensem_model.hmean(pred)
            for x, y in zip(ensem_pred.data.tolist(), gold.data.tolist()):
                for m, n in zip(x, y):
                    self.assertEqual(utils.myRound(m), utils.myRound(n))
        print("test harmonic mean method")


if __name__ == "__main__":
    unittest.main()
