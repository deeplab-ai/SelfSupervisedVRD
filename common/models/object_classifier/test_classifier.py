# -*- coding: utf-8 -*-
"""Object Classification module for multiple SGG datasets."""
import sys

from torch import nn
import torch
from os import path as osp
from .base_object_classifier import BaseObjectClassifier
from .object_classifier import ObjectClassifier
from ..object_classifier.transformer_classifier import TransformerClassifier


class TestClassifier(BaseObjectClassifier):
    """Object Classifier based on Faster-RCNN."""

    def __init__(self, config):
        """Initialize class using checkpoint."""
        super().__init__(config, {'pool_features'})

        self.logger = config.logger
        self.projection_head = config.projection_head
        self.num_obj_dim = config.num_obj_dim
        self.test_on_masked = config.test_on_masked
        self.pretrained_net = TransformerClassifier(config, return_only_scores=False, mask=self.test_on_masked)
        pretrained_model = config.pretrained_model
        tmp_path_name = f"prerequisites/models/{pretrained_model}_preddet_VG200_finetuned"
        self.logger.debug(f'Loading {tmp_path_name.split("/")[-1]} model..')
        self.logger.warn("The pretrained models are hard coded for VG200 dataset only!!")
        tmp_path_name = osp.join(tmp_path_name, 'model.pt')
        checkpoint = torch.load(tmp_path_name, map_location=self._device)
        self.pretrained_net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        for param in self.pretrained_net.parameters():
            param.requires_grad = False
        self.pretrained_net.to(self._device)

        self.obj_cls = ObjectClassifier(config)
        pretrained_model = 'obj_cls_256' if config.dataset == 'VG200' else 'obj_cls_256_v3'
        tmp_path_name = f"prerequisites/models/{pretrained_model}_preddet_{config.dataset}_finetuned"
        self.logger.debug(f'Loading {tmp_path_name.split("/")[-1]} model..')

        tmp_path_name = osp.join(tmp_path_name, 'model.pt')
        checkpoint = torch.load(tmp_path_name, map_location=self._device)
        self.obj_cls.load_state_dict(checkpoint['model_state_dict'], strict=False)
        for param in self.obj_cls.parameters():
            param.requires_grad = False
        self.obj_cls.to(self._device)

        self.l1 = self.obj_cls.l1
        self.relu = self.obj_cls.relu
        self.l2 = self.obj_cls.l2

    def net_forward(self, base_features, objects):
        """Forward pass, override."""
        pretrained_out = self.pretrained_net(
            objects['input_image'],
            objects['input_object_boxes'],
            objects['input_object_ids'],
            objects['pairs'],
            objects['input_image_info']
        )

        return self._forward(pretrained_out[2], pretrained_out[3])

    def _forward(self, features, mask_idx_obj):
        """Forward pass."""
        return self.l2(self.relu(self.l1(features))), mask_idx_obj
