# -*- coding: utf-8 -*-
"""Finetune Network, by Anastasakis et al., 2024"""

import torch
from torch import nn
from os import path as osp

from .base_sg_generator import BaseSGGenerator
from ..object_classifier.transformer_classifier import TransformerClassifier
from .mbbr import MBBR


class Mlp_finetune(BaseSGGenerator):
    """Extends PyTorch nn.Module."""

    def __init__(self, config):
        """Initialize layers."""
        super().__init__(config, {'base_features', 'pool_features'})

        self.logger = config.logger
        self.pretrain_task = config.pretrain_task
        self.projection_head = config.projection_head
        assert not config.mask_before_frcnn, "mask_before_frcnn must be disabled."
        assert self.pretrain_task == 'classification' or config.pretrain_task == 'reconstruction', "Please configure a valid pretrain task: 'classification' or 'reconstruction'"
        if self.pretrain_task == 'classification':
            self.logger.debug(f"Pretrain task: {self.pretrain_task}")
            self.pretrained_net = TransformerClassifier(config, return_only_scores=False, mask=False)
        else:
            self.logger.debug(f"Pretrain task: {self.pretrain_task}")
            self.pretrained_net = MBBR(config, mask=False)  # no masking

        if config.finetune:
            self.pretrained_net = MBBR(config, mask=False)
            model_path_name = config.paths['models_path']
            self.logger.debug('Loading pretrained model... ')
            tmp_path_name = model_path_name.split('finetuned')[0] + 'pretrained'
        elif config.pretrained_model is not None:
            pretrained_model = config.pretrained_model
            task = config.task
            dataset = config.dataset
            mode = 'pretrained' if self.pretrain_task == 'reconstruction' else 'finetuned'
            tmp_path_name = config.prerequisites_path + f"models/{pretrained_model}_preddet_VG200_{mode}"
            self.logger.debug(f'Loading {tmp_path_name.split("/")[-1]} model..')
            self.logger.warn("The pretrained models are hard coded for VG200 dataset only!!")
        else:
            raise AttributeError("Something went wrong with the finetune")
        tmp_path_name = osp.join(tmp_path_name, 'model.pt')
        checkpoint = torch.load(tmp_path_name, map_location=self._device)
        self.pretrained_net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        for param in self.pretrained_net.parameters():
            param.requires_grad = False

        self.pretrained_net.to(self._device)

        # Classification
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64 + 128, 512), nn.ReLU(),
            nn.Linear(512, self.num_rel_classes)
        )
        self.bin_classifier = nn.Sequential(
            nn.Linear(128 + 64 + 128, 512), nn.ReLU(),
            nn.Linear(512, 2)
        )

        # Spatial features
        self.delta_fusion = nn.Sequential(
            nn.Linear(38, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )

        # Language features
        self.fc_subject_lang = nn.Sequential(nn.Linear(300, 256), nn.ReLU())
        self.fc_object_lang = nn.Sequential(nn.Linear(300, 256), nn.ReLU())
        self.lang_fusion = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )

        # Visual features from the transformer
        self.visual_fusion = nn.Sequential(nn.Linear(256 * 2, 128), nn.ReLU())

        # weights init
        self.logger.debug("Initializing weights..")
        # classifier
        torch.nn.init.xavier_uniform_(self.classifier[0].weight)
        torch.nn.init.xavier_uniform_(self.classifier[2].weight)
        # delta fusion
        torch.nn.init.xavier_uniform_(self.delta_fusion[0].weight)
        torch.nn.init.xavier_uniform_(self.delta_fusion[2].weight)
        # lang fusion
        torch.nn.init.xavier_uniform_(self.lang_fusion[0].weight)
        torch.nn.init.xavier_uniform_(self.lang_fusion[2].weight)
        # visual fusion
        torch.nn.init.xavier_uniform_(self.visual_fusion[0].weight)

    # prepare features (arguments) for _forward
    def net_forward(self, base_features, objects, pairs):
        """Forward pass, override."""
        if self.pretrain_task == 'reconstruction':
            pretrained_out = self.pretrained_net(
                objects['input_image'],
                objects['input_object_boxes'],
                objects['input_object_ids'],
                pairs,
                objects['input_image_info']
            )
        else:
            pretrained_out = self.pretrained_net(
                objects['input_image'],
                objects['input_object_boxes'],
                objects['input_object_ids'],
                pairs,
                objects['input_image_info']
            )

        return self._forward(
            pretrained_out[2],
            pretrained_out[0],
            pairs,
            self.get_obj_embeddings(objects['ids'][pairs[:, 0]]),
            self.get_obj_embeddings(objects['ids'][pairs[:, 1]]),
            self.get_spatial_features(
                objects['boxes'][pairs[:, 0]],
                objects['boxes'][pairs[:, 1]],
                method='gkanatsios_2019b'
            )
        )

    def _forward(self, transformer_output, output, pairs, subj_embs, obj_embs,
                 spat_feats):
        """Forward pass, returns output scores."""
        spat_features = self.spatial_forward(spat_feats)
        lang_features = self.language_forward(subj_embs, obj_embs)

        if self.projection_head or self.pretrain_task == 'classification':
            output = transformer_output

        # transformer/mae features
        subjects = output[pairs[:, 0]]  # only visuals
        objects = output[pairs[:, 1]]  # only visuals

        vis_features = self.visual_fusion(
            torch.cat((subjects, objects), dim=1)
        )
        scores = self.classifier(
            torch.cat((vis_features, spat_features, lang_features), dim=1)
        )
        bin_scores = self.bin_classifier(
            torch.cat((vis_features, spat_features, lang_features), dim=1)
        )

        if self.mode == 'test':
            scores = self.softmax(scores)
            bin_scores = self.softmax(bin_scores)

        return scores, bin_scores, transformer_output, output

    def language_forward(self, subj_embs, obj_embs):
        """Forward of language net."""
        embeddings = torch.cat(
            (self.fc_subject_lang(subj_embs), self.fc_object_lang(obj_embs)),
            dim=1
        )
        features = self.lang_fusion(embeddings)
        return features

    def spatial_forward(self, spat_feats):
        """Forward of spatial net."""
        features = self.delta_fusion(spat_feats)
        return features
