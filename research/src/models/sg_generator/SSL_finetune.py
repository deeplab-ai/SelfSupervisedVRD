# -*- coding: utf-8 -*-
"""A baseline using only linguistic and spatial features."""
import torch

from common.models.sg_generator.SSL_finetune import Mlp_finetune
from research.src.train_testers import SGGTrainTester


class TrainTester(SGGTrainTester):
    """Extends SGGTrainTester."""

    def __init__(self, net, config, features, obj_classifier, teacher):
        """Initialize instance."""
        super().__init__(net, config, features, obj_classifier, teacher)

    def _compute_loss(self, batch, step):
        """Compute loss for current batch."""
        # Net outputs and targets
        outputs = self._net_forward(batch, step)[0]
        targets = self.data_loader.get('predicate_ids', batch, step)

        losses = {
            'CE': self.criterion(outputs, targets),  # why tensor?
        }
        # Losses
        # loss = losses['CE']
        loss = losses['CE']

        return loss, losses


def train_test(config, obj_classifier=None, teacher=None):
    """Train and test a model."""
    # pretrained_net = TransformerTest(config)
    net = Mlp_finetune(config)
    train_tester = TrainTester(net, config, {'images'}, obj_classifier, teacher)
    train_tester.train_test()
