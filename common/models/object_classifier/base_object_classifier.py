# -*- coding: utf-8 -*-
"""A class to be inherited by other object classifiers."""
import sys

from common.models.sg_generator import BaseSGGenerator


class BaseObjectClassifier(BaseSGGenerator):
    """
    Extends PyTorch nn.Module, base class for object classifiers.

    Inputs:
        - config: Config object, see config.py
        - features: set of str, features computed on-demand:
            - object_1hots: object 1-hot vectors
            - object_masks: object binary masks
            - pool_features: object pooled features (vectors)
            - roi_features: object pre-pooled features (volumes)
    """

    def __init__(self, config, features):
        """Initialize layers."""
        super().__init__(config, features)
        self.pred_top_net = None
        self.mask_before_frcnn = config.mask_before_frcnn

    def forward(self, image, object_boxes, object_ids, pairs, image_info):
        """
        Forward pass.

        Expects:
            - image: image tensor, (3 x H x W) or None
            - object_boxes: tensor, (n_obj, 4), (xmin, ymin, xmax, ymax)
            - object_ids: tensor, (n_obj,), or None, object category ids
            - image_info: tuple, (im_width, im_height)
        """
        # Base features
        self._image_info = image_info
        mask_idx_obj = None
        if self.mode == 'train' and self.mask_before_frcnn:
            # check if this if statement is needed
            print("Wrong")
            image, mask_idx_obj = self.create_mask_object(image, object_boxes)
        base_features = self.get_base_features(image)
        # Object features
        # objects = {'boxes': object_boxes, 'ids': object_ids, 'mask_idx_obj': mask_idx_obj}
        objects = {'boxes': object_boxes, 'ids': object_ids, 'mask_idx_obj': mask_idx_obj, 'input_image': image,
                   'input_object_boxes': object_boxes, 'input_object_ids': object_ids, 'input_image_info': image_info,
                   'pairs': pairs}
        if 'pool_features' in self.features:
            objects['pool_features'] = self.get_obj_pooled_features(
                base_features, object_boxes
            )
        if 'roi_features' in self.features:
            objects['roi_features'] = self.get_roi_features(
                base_features, object_boxes
            )
        if 'object_1hots' in self.features:
            objects['1hots'] = self.get_obj_1hot_vectors(object_ids)
        if 'object_masks' in self.features:
            objects['masks'] = self.get_obj_masks(object_boxes)
        # Forward pass
        return self.net_forward(base_features, objects)

    @staticmethod
    def net_forward(base_features, objects):
        """Forward pass, override."""
        return []
