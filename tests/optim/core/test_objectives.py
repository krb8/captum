#!/usr/bin/env python3
import unittest

import numpy as np
import torch.nn as nn

import captum.optim as opt
from tests.helpers.basic import BaseTest


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        return self.layer(x)


class TestInputOptimization(BaseTest):
    def test_input_optimization(self) -> None:
        model = SimpleModel()
        loss_fn = opt.loss.ChannelActivation(model.layer, 0)
        obj = opt.InputOptimization(model, loss_function=loss_fn)
        history = obj.optimize(opt.objectives.n_steps(5))
        history = np.mean(np.mean(history, axis=2), axis=2)
        for i, j in zip(history[:-1], history[1:]):
            assert i < j


if __name__ == "__main__":
    unittest.main()
