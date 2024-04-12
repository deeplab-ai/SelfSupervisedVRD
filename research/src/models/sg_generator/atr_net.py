# -*- coding: utf-8 -*-
"""Attention-Translation-Relation Network, Gkanatsios et al., 2019."""

import torch
import sys
from torch.nn import functional as F

from common.models.sg_generator import ATRNet
from research.src.train_testers import SGGTrainTester


class TrainTester(SGGTrainTester):
    """Extends SGGTrainTester."""

    def __init__(self, net, config, features, obj_classifier, teacher):
        """Initialize instance."""
        super().__init__(net, config, features, obj_classifier, teacher)
        self.batch_counter = 0
        self.softmax = torch.nn.Softmax(dim=1)
        self.thres = config.thres
        self.mse = torch.nn.MSELoss()

    def _compute_loss(self, batch, step):
        """Compute loss for current batch."""
        '''
        VG200: 60948 images with 425.066 labeled relations, atr trained with 63972 labeled relations
        VRD: 3773 images with 203280 relations, atr trained with 19779 labeled relations
        '''
        # Net outputs and targets
        if self.lb_imgs and not self.thres:
            raise AttributeError("Please configure a threshold for fix match loss")

        outputs = self._net_forward(batch, step)
        scores, p_scores, os_scores = (outputs[0], outputs[2], outputs[3])
        # print(scores.shape)
        # print(os_scores.shape)
        if not self.lb_imgs:
            targets = self.data_loader.get('predicate_ids', batch, step)

            losses = {
                'CE': self.criterion(scores, targets),
                'p-CE': self.criterion(p_scores, targets),
                'os-CE': self.criterion(os_scores, targets)
            }

            loss = losses['CE'] + losses['p-CE'] + losses['os-CE']
        else:
            '''
            CUDA_VISIBLE_DEVICES=3 python3 main_research.py --model=atr_fix_match_15lambda_0.9 --net_name=test --random_seed=4 --lb_imgs=500 --normal
            '''

            targets = batch['predicate_ids'][step].to(self._device)
            labeled = batch['labeled'][step]

        # num_rel = targets.size(0)
        # num_labeled = int(0.1 * targets.size(0)) if 0.1 * targets.size(0) >= 1 else 1
        # num_unlabeled = targets.size(0) - num_labeled
        # print(f"{num_labeled}, {num_unlabeled} out of {targets.size(0)}")
            if labeled == 0:
                os_scores_targets = self.softmax(os_scores.detach())
                # print(os_scores_targets)
                c = torch.Tensor(
                    [[True if os_scores_targets[j][i] >= self.thres else False for i in range(os_scores_targets.size(1))] for j in
                     range(os_scores_targets.size(0))]).bool().to(self._device)
                idxs = (torch.any(c, 1).long() == 1).nonzero(as_tuple=False)[:, 0]
                # idxs += num_labeled
                os_scores_targets = torch.argmax(os_scores_targets, dim=1)

                losses = {
                    'CE': torch.zeros(1).to(self._device),
                    'p-CE': torch.zeros(1).to(self._device),
                    'os-CE': torch.zeros(1).to(self._device),
                    # 'p_os-CE': self.criterion(p_scores[idxs], os_scores_targets[idxs])
                    'p_os-CE': self.mse(p_scores, os_scores)
                }

            else:
                losses = {
                    'CE': self.criterion(scores, targets),
                    'p-CE': self.criterion(p_scores, targets),
                    'os-CE': self.criterion(os_scores, targets),
                    'p_os-CE': torch.zeros(1).to(self._device)
                }

            loss = losses['CE'] + losses['p-CE'] + losses['os-CE'] + 10 * losses['p_os-CE']

        # losses = {
        #     'CE': self.criterion(scores[:num_labeled], targets[:num_labeled]),
        #     'p-CE': self.criterion(p_scores[:num_labeled], targets[:num_labeled]),
        #     'os-CE': self.criterion(os_scores[:num_labeled], targets[:num_labeled]),
        #     'p_os-CE': self.criterion(p_scores[-num_unlabeled:], os_scores_targets[-num_unlabeled:])
        # }
        # losses = {
        #     'CE': torch.cat((self.criterion(scores[:num_labeled], targets[:num_labeled]),
        #                      torch.zeros(num_unlabeled).to(self._device)), dim=0),
        #     'p-CE': torch.cat((self.criterion(p_scores[:num_labeled], targets[:num_labeled]),
        #                        torch.zeros(num_unlabeled).to(self._device)), dim=0),
        #     'os-CE': torch.cat((self.criterion(os_scores[:num_labeled], targets[:num_labeled]),
        #                         torch.zeros(num_unlabeled).to(self._device)), dim=0),
        #     'p_os-CE': torch.cat((torch.zeros(num_labeled + num_unlabeled - idxs.size(0)).to(self._device),
        #                           self.criterion(p_scores[-num_unlabeled:][idxs], os_scores_targets[idxs])),
        #                          dim=0) if idxs.size(0) > 0 else torch.zeros(num_rel).to(self._device)
        # }

        # loss = losses['CE'] + losses['p-CE'] + losses['os-CE'] + 15 * losses['p_os-CE']
        # loss = losses['CE'] + losses['p-CE'] + losses['os-CE']

        if self._use_multi_tasking and self._task != 'preddet':
            bg_scores = outputs[1]
            bg_p_scores = outputs[4]
            bg_os_scores = outputs[5]
            bg_targets = self.data_loader.get('bg_targets', batch, step)
            loss = (
                    loss
                    + F.cross_entropy(bg_scores, bg_targets, reduction='none')
                    + F.cross_entropy(bg_p_scores, bg_targets, reduction='none')
                    + F.cross_entropy(bg_os_scores, bg_targets, reduction='none')
            )
        if self.teacher is not None and self._negative_loss is None \
                and not self._use_consistency_loss:
            losses['KD'] = self._kd_loss(scores, outputs[1], batch, step)
            if self.training_mode:
                loss += losses['KD']
        if self._negative_loss is not None:
            if self._neg_classes is not None:
                neg_classes = self._neg_classes
            else:
                neg_classes = [i for i in range(self.net.num_rel_classes - 1)]
            losses['NEG'] = self._negatives_loss(
                outputs, batch, step, neg_classes, self._negative_loss)
            if self.training_mode:
                loss += losses['NEG']
        if self._use_consistency_loss and self._epoch >= 1:
            cons_loss = \
                self._consistency_loss(batch, step, scores, typ='triplet_sm')
            losses['Cons'] = cons_loss
            if self.training_mode:
                loss += cons_loss
        return loss, losses


def train_test(config, obj_classifier=None, teacher=None):
    """Train and test a net."""
    net_params = config_net_params(config)
    net = ATRNet(
        config,
        attention=net_params['attention'],
        use_language=net_params['use_language'],
        use_spatial=net_params['use_spatial'])
    train_tester = TrainTester(
        net, config, {'images'}, obj_classifier, teacher
    )
    train_tester.train_test()


def config_net_params(config):
    """Configure net parameters."""
    net_params = {
        'attention': 'multi_head',
        'use_language': True,
        'use_spatial': True
    }
    if 'single_head' in config.net_name:
        net_params['attention'] = 'single_head'
    if 'no_att' in config.net_name:
        net_params['attention'] = None
    if 'no_lang' in config.net_name:
        net_params['use_language'] = False
    if 'no_spat' in config.net_name:
        net_params['use_spatial'] = False
    return net_params
