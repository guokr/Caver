import unittest
from caver.evaluator import *
import torch.nn as nn
import torch


class TestEvaluator(unittest.TestCase):
    def test_evaluate(self):
        criterion = nn.BCEWithLogitsLoss()
        evaluator = Evaluator(criterion)
        preds = torch.Tensor([[0.9, 0.8, 0.7, 0.6, 0.5]])
        target = torch.Tensor([[1, 0, 0, 0, 0]])
        _, recall, precision, f_score = evaluator.evaluate(preds, target, "test")
        self.assertEqual(1.0, recall)
        self.assertEqual(0.273761868228963, precision)
        self.assertEqual(0.4298477997454912, f_score)
        print("test evaluate")


if __name__ == "__main__":
    unittest.main()
