# -*- coding: utf-8 -*-
"""Object Classification module for multiple SGG datasets."""

from common.models.object_classifier.transformer_classifier import TransformerClassifier
from research.src.train_testers import ObjClsTrainTester


def train_test(config, obj_classifier=None, teacher=None):
    """Train and test a net."""
    net = TransformerClassifier(config)
    train_tester = ObjClsTrainTester(net, config, set(), teacher)
    train_tester.train_test()
