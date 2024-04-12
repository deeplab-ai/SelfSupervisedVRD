# -*- coding: utf-8 -*-
import torch

from common.models.sg_generator.mbbr import MBBR
from research.src.train_testers import SGGTrainTester


class TrainTester(SGGTrainTester):
    """Extends SGGTrainTester."""

    def __init__(self, net, config, features, obj_classifier, teacher):
        """Initialize instance."""
        super().__init__(net, config, features, obj_classifier, teacher)
        # self.rec_loss = torch.nn.MSELoss()
        self.use_cos_loss = config.cosine_loss
        self.rec_loss = torch.nn.CosineSimilarity() if config.cosine_loss else torch.nn.MSELoss()
        msg = 'Using cosine similarity..' if config.cosine_loss else 'Using MSE loss..'
        self.logger.debug(msg)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        # p = self.patch_embed.patch_size[0]
        p = 16
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def _compute_loss(self, batch, step):
        """Compute loss for current batch."""
        # Net outputs and targets
        outputs = self._net_forward(batch, step)
        targets = self.data_loader.get('predicate_ids', batch, step)
        predictions = outputs[0]

        targets = outputs[1]
        size = outputs[-1].shape[0]

        losses = {
            'reconstruction_loss': ((predictions - targets) ** 2).mean(dim=-1)
        }
        loss = losses['reconstruction_loss']
        return loss, losses


def train_test(config, obj_classifier=None, teacher=None):
    """Train and test a model."""
    net = MBBR(config)
    train_tester = TrainTester(net, config, {'images'}, obj_classifier, teacher)
    train_tester.train_test()
