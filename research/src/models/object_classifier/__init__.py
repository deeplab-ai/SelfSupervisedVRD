# -*- coding: utf-8 -*-
"""Object classifiers."""

import torch

from common.models.object_classifier import ObjectClassifier


def load_classifier(name, config):
    """Load trained classifier for given name and config."""
    # TODO: change required path for object classifier
    if name == 'object_classifier':
        obj_classifier = ObjectClassifier(config)
    checkpoint = torch.load(
        config.paths['models_path'] + name
        + '_objcls_' + (config.dataset if config.dataset != 'UnRel' else 'VRD')
        + '.pt',
        map_location=config.device)
    obj_classifier.load_state_dict(checkpoint['model_state_dict'])
    return obj_classifier.to(config.device)
