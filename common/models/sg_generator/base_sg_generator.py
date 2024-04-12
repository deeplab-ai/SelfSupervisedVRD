# -*- coding: utf-8 -*-
"""A class to be inherited by other scene graph generators."""
import sys
from copy import deepcopy
from pdb import set_trace
import json

import numpy as np
import torch
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import resnet50
from torchvision.ops import MultiScaleRoIAlign, roi_align, roi_pool

from common.tools import SpatialFeatureExtractor
# from common.models.sg_generator.Transformer_Test import TransformerTest


class BaseSGGenerator(nn.Module):
    """
    Extends PyTorch nn.Module, base class for generators.

    Inputs:
        - config: Config object, see config.py
        - features: set of str, features computed on-demand:
            - base_features: backbone conv. features
            - object_1hots: object 1-hot vectors
            - object_masks: object binary masks
            - pool_features: object pooled features (vectors)
            - roi_features: object pre-pooled features (volumes)
    """

    def __init__(self, config, features, **kwargs):
        """Initialize layers."""
        super().__init__()
        self.reset_from_config(config)
        self.features = features
        self._mask_size = kwargs.get('mask_size', 32)

        # Visual backbone
        resnet50_backbone = resnet50(pretrained=True)
        _backbone = fasterrcnn_resnet50_fpn(pretrained=True)
        for name, param in _backbone.named_parameters():
            if name.startswith('backbone'):
                param.requires_grad = False
            elif name.startswith('roi_heads.box_head'):
                param.requires_grad = self.train_top
        for param in resnet50_backbone.parameters():
            param.requires_grad = False
        self.resnet50_backbone = resnet50_backbone
        self.res_image = None
        self.backbone = _backbone.backbone
        self.box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=14,
            sampling_ratio=2)
        self.orig_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2)
        # self.test_roi_pool = MultiScaleRoIAlign(
        #     featmap_names=['0', '1', '2', '3'],
        #     output_size=self._mask_size,
        #     sampling_ratio=2)
        _top_net = nn.Sequential(
            _backbone.roi_heads.box_head.fc6, nn.ReLU(),
            _backbone.roi_heads.box_head.fc7, nn.ReLU())
        self.obj_top_net = deepcopy(_top_net)
        self.pred_top_net = deepcopy(_top_net)
        self.transform = _backbone.transform

        # Depth network
        if 'depth' in self.features:
            self._depth_net = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self._depth_net.eval()
            for p in self._depth_net.parameters():
                p.requires_grad = False
            self._depth_transform = torch.hub.load(
                "intel-isl/MiDaS", "transforms").small_transform
            self._depth_size = 16

        # Rest parameters
        self.spatial_extractor = SpatialFeatureExtractor()
        self.softmax = nn.Softmax(dim=1)
        self.mode = 'train'

    def forward(self, image, object_boxes, object_ids, pairs, image_info):
        """
        Forward pass.

        Expects:
            - image: image tensor, (3 x H x W) or None
            - object_boxes: tensor, (n_obj, 4), (xmin, ymin, xmax, ymax)
            - object_ids: tensor, (n_obj,), object category ids
            - pairs: tensor, (n_rel, 2), pairs of objects to examine
            - image_info: tuple, (im_width, im_height)
        """
        # Base features
        self._image_info = image_info
        base_features = None
        mask_idx_obj = None
        # if self.mode == 'train' and self.mask_before_frcnn:
        if self.mask_before_frcnn:
            image, mask_idx_obj = self.create_mask_object(image, object_boxes)
        if 'base_features' in self.features:
            base_features = self.get_base_features(image)
        # Object features
        objects = {'boxes': object_boxes, 'ids': object_ids, 'mask_idx_obj': mask_idx_obj, 'input_image': image,
                   'input_object_boxes': object_boxes, 'input_object_ids': object_ids, 'input_image_info': image_info}
        if 'pool_features' in self.features:
            objects['pool_features'] = self.get_obj_pooled_features(
                base_features, object_boxes
            )
        # if pretrained_net is not None:
        #     objects['pretrained_out'] =
        if 'roi_features' in self.features:
            objects['roi_features'] = self.get_roi_features(
                base_features, object_boxes
            )
        if 'object_1hots' in self.features:
            objects['1hots'] = self.get_obj_1hot_vectors(object_ids)
        if 'object_masks' in self.features:
            objects['masks'] = self.get_obj_masks(object_boxes)
        # Refine object features using context
        objects = self.contextualize(objects, base_features)
        # Depth features
        if 'depth' in self.features:
            img_d = self._depth_transform(
                image.permute(1, 2, 0).cpu().numpy()
            ).to(self._device)
            objects['depth'] = self._depth_net(img_d)
            self._box_scales_d = self._compute_scales(
                image.shape[-2:], objects['depth'].shape[-2:])

        # Iterative forward pass over sub-batches for memory issues
        outputs = [
            self.net_forward(
                base_features, objects,
                pairs[range(
                    btch * self.rel_batch_size,
                    min((btch + 1) * self.rel_batch_size, len(pairs))
                )]
            )
            for btch in range(1 + (len(pairs) - 1) // self.rel_batch_size)
        ]
        #print(len(outputs))
        # res = [
        #     torch.cat([output[k] for output in outputs], dim=0)
        #     if outputs[0][k] is not None else None
        #     for k in range(len(outputs[0]))
        # ]
        # print(len(res))
        # sys.exit(2)
        return [
            torch.cat([output[k] for output in outputs], dim=0)
            if outputs[0][k] is not None else None
            for k in range(len(outputs[0]))
        ]

    # @staticmethod
    def create_mask_object(self, image, object_boxes):
        """
        Mask objects before Faster-RCNN with 15% probability
        to prevent information leakage

        image: tensor, (3 x H x W)
        object_boxes: tensor, (n_obj, 4), (xmin, ymin, xmax, ymax)

        returns
        masked_img: tensor, the original image with one object randomly masked.
        """
        # torch.set_printoptions(threshold=1000000)
        num_objs = object_boxes.shape[0]
        # masked_bbox = torch.randint(num_objs, (1,)).item()
        masked_obj_ids = torch.rand(num_objs)
        masked_obj_ids = masked_obj_ids < 0.15  # mask each object with 15% probability
        # bbox = object_boxes[masked_bbox].int()
        # image[:, bbox[1] - 1:bbox[3], bbox[0] - 1:bbox[2]] = 0

        masked_bboxes = object_boxes[masked_obj_ids].int()
        for masked_bbox in masked_bboxes:
            image[:, masked_bbox[1] - 1:masked_bbox[3], masked_bbox[0] - 1:masked_bbox[2]] = 0

        return image, masked_obj_ids

    @staticmethod
    def contextualize(objects, base_features):
        """Refine object features."""
        return objects  # no-context case, re-implement else

    @staticmethod
    def net_forward(base_features, objects, pairs):
        """Forward pass, override."""
        return [], []

    @torch.no_grad()
    def get_base_features(self, image):
        """Forward pass for a list of image tensors."""
        orig_shape = image.shape[-2:]
        image, _ = self.transform([image], None)
        self.res_image = image
        self._img_shape = image.image_sizes[0]
        self._box_scales = self._compute_scales(orig_shape, self._img_shape)
        return self.backbone(image.tensors)

    @torch.no_grad()
    def get_resnet50_features(self):
        """Forward pass for a list of image tensors."""
        # orig_shape = image.shape[-2:]
        # image, _ = self.transform([image], None)
        # self._img_shape = image.image_sizes[0]
        # self._box_scales = self._compute_scales(orig_shape, self._img_shape)
        return self.resnet50_backbone(self.res_image.tensors)

    def get_obj_1hot_vectors(self, object_ids):
        """Forward pass for a list of object ids."""
        obj_vecs = self.obj_zeros[[0] * len(object_ids)]
        obj_vecs[torch.arange(len(object_ids)), object_ids] = 1.0
        return obj_vecs

    def get_obj_embeddings(self, object_ids):
        """Forward pass for a list of object ids."""
        if self.obj2vec is None:
            self._set_word2vec()
        return self.obj2vec[object_ids]

    def get_obj_masks(self, object_boxes, mask_size=None):
        """Forward pass for a list of object boxes."""
        return self.spatial_extractor.get_binary_masks(
            object_boxes, self._image_info[1], self._image_info[0],
            self._mask_size if mask_size is None else mask_size
        ).to(self._device)

    def get_obj_pooled_features(self, base_features, rois):
        """Forward pass for a list of object rois."""
        rois = self._rescale_boxes(rois, self._box_scales)
        features = self.orig_roi_pool(base_features, [rois], [self._img_shape])
        # test_features = self.test_roi_pool(base_features, [rois], [self._img_shape])
        # return self.obj_top_net(features.flatten(start_dim=1)), test_features
        return self.obj_top_net(features.flatten(start_dim=1))

    def get_pred_embeddings(self, predicate_ids):
        """Forward pass for a list of predicate ids."""
        if self.pred2vec is None:
            self._set_word2vec()
        return self.pred2vec[predicate_ids]

    def get_pred_pooled_features(self, base_features, subj_rois, obj_rois):
        """Forward pass for a list of rois."""
        subj_rois = self._rescale_boxes(subj_rois, self._box_scales)
        obj_rois = self._rescale_boxes(obj_rois, self._box_scales)
        rois = self._create_pred_boxes(subj_rois, obj_rois)
        features = self.orig_roi_pool(base_features, [rois], [self._img_shape])
        return self.pred_top_net(features.flatten(start_dim=1))

    def get_pred_probabilities(self, subj_ids, obj_ids):
        """Forward pass for a list of subject and object ids."""
        if self.probabilities is None:
            self._set_probabilities()
        return self.probabilities[subj_ids, obj_ids]

    def get_roi_features(self, base_features, rois, rois2=None):
        """Forward pass for a list of rois."""
        rois = self._rescale_boxes(rois, self._box_scales)
        if rois2 is not None:  # consider pair of subj-obj boxes
            rois2 = self._rescale_boxes(rois2, self._box_scales)
            rois = self._create_pred_boxes(rois, rois2)
        return self.box_roi_pool(base_features, [rois], [self._img_shape])

    def get_depth_features(self, depth_features, rois, rois2=None):
        rois = self._rescale_boxes(rois, self._box_scales_d)
        if rois2 is not None:  # consider pair of subj-obj boxes
            rois2 = self._rescale_boxes(rois2, self._box_scales_d)
            rois = self._create_pred_boxes(rois, rois2)
        return roi_align(depth_features, [rois], self._depth_size)

    def get_spatial_features(self, subj_boxes, obj_boxes,
                             method='gkanatsios_2019b', depth=None):
        """Forward pass for a list of object boxes."""
        return self.spatial_extractor.get_features(
            subj_boxes, obj_boxes,
            self._image_info[1], self._image_info[0], method, depth
        ).to(self._device)

    def reset_from_config(self, config):
        """Reset parameters from a config object."""
        self.finetune = config.finetune
        self._device = config.device
        self.dataset = config.dataset
        self.json_path = config.paths['json_path']
        self.num_obj_classes = config.num_obj_classes
        self.num_rel_classes = config.num_rel_classes
        obj_zeros = torch.zeros(self.num_obj_classes).unsqueeze(0)
        self.obj_zeros = obj_zeros.to(self._device)
        self.obj2vec = None
        self.pred2vec = None
        self.probabilities = None
        self.rel_batch_size = config.rel_batch_size
        self.train_top = config.train_top
        self.use_coco = config.use_coco
        self.mask_before_frcnn = config.mask_before_frcnn

    def save_memory(self):
        """Nullify unnecessary components to save memory."""
        self.backbone = None
        self.pred_top_net = None
        self.obj_top_net = None

    def train(self, mode=True):
        """Override train to prevent modules from being trainable."""
        for name, child in self.named_children():
            if name.startswith('backbone'):
                child.eval()  # always on eval to disable batch normalization
            elif 'top_net' in name:
                # https://github.com/torch/nn/issues/873
                child.train(mode=mode and self.train_top)
            else:
                child.train(mode=mode)

    @staticmethod
    def _compute_scales(orig_shape, new_shape):
        """Compute per dimension scaling factor for image rois."""
        return [ndim / odim for ndim, odim in zip(new_shape, orig_shape)]

    def _create_pred_boxes(self, subj_rois, obj_rois):
        """Create predicate boxes given subject and object boxes."""
        return self.spatial_extractor.create_pred_boxes(subj_rois, obj_rois)

    def _create_3d_boxes(self, boxes, dmap):
        """Augment 2d boxes with the depth dimension"""
        boxes = self._rescale_boxes(boxes, self._box_scales_d)
        z_max = roi_pool(dmap.unsqueeze(0), [boxes], 1).view(-1, 1)
        z_min = -roi_pool(-dmap.unsqueeze(0), [boxes], 1).view(-1, 1)
        boxes = torch.cat((boxes[:, :2], z_min, boxes[:, 2:], z_max), dim=1)
        return boxes

    @staticmethod
    def _rescale_boxes(rois, scales):
        """Rescale rois to match the resized image dimensions."""
        ratio_height, ratio_width = scales
        xmin, ymin, xmax, ymax = rois.unbind(1)
        xmin = xmin * ratio_width
        xmax = xmax * ratio_width
        ymin = ymin * ratio_height
        ymax = ymax * ratio_height
        return torch.stack((xmin, ymin, xmax, ymax), dim=1)

    def _set_probabilities(self):
        """Set predicate probability matrix for given dataset."""
        json_name = self.json_path + self.dataset + '_probabilities.json'
        with open(json_name) as fid:
            probs = torch.from_numpy(np.array(json.load(fid))).float()
        self.probabilities = probs.to(self._device)

    def _set_word2vec(self):
        """Load dataset word2vec array."""
        w2v_json = self.json_path + self.dataset + '_word2vec.json'
        with open(w2v_json) as fid:
            w2vec = json.load(fid)  # word2vec dictionary
            obj2vec = torch.from_numpy(np.array(w2vec['objects'])).float()
            p2vec = torch.from_numpy(np.array(w2vec['predicates'])).float()
        if self.use_coco:
            with open(self.json_path + 'COCO_word2vec.json') as fid:
                w2vec = json.load(fid)  # word2vec dictionary
                obj2vec = torch.from_numpy(np.array(w2vec['objects'])).float()
        self.obj2vec = obj2vec.to(self._device)
        self.pred2vec = p2vec.to(self._device)
