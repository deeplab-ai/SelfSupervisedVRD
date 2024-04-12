# -*- coding: utf-8 -*-
"""Object Classification module for multiple SGG datasets."""

from torch import nn
import torch
import math
from torchvision.ops import MultiScaleRoIAlign
from .base_object_classifier import BaseObjectClassifier
from common.models.transformers.base_transformer import TransformerDecoder, TransformerDecoderLayer
from common.models.transformers.base_transformer import TransformerEncoder, TransformerEncoderLayer


class TransformerClassifier(BaseObjectClassifier):
    """Object Classifier based on Faster-RCNN."""

    def __init__(self, config, return_only_scores=True, mask=True):
        """Initialize class using checkpoint."""
        super().__init__(config, {'pool_features'})
        # return_only_scores is True ONLY when the classifier is trained else False

        self.logger = config.logger
        self.device = config._device
        self.return_only_scores = return_only_scores
        self.num_obj_dim = config.num_obj_dim
        self.test_on_masked = config.test_on_masked

        self.mask = mask if not config.mask_before_frcnn else False
        if self.test_on_masked:
            self.mask = True
        self.add_pos_emb = PositionalEmbedding(d_model=256)
        self.obj_pos = nn.Linear(512, 256)
        self.obj_classifier = nn.Linear(256, self.num_obj_classes if return_only_scores else 150)

        assert config.pretrain_arch == 'encoder' or config.pretrain_arch == 'decoder', "Please configure a valid transformer architecture: 'encoder' or 'decoder'"
        self.pretrain_arch = config.pretrain_arch
        self.logger.debug(f"Pretrain architecture: {self.pretrain_arch}.")
        if self.pretrain_arch == 'decoder':
            self.decoder_layer = TransformerDecoderLayer(d_model=256, nhead=8)
            self.transformer = TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=6, residual=False)
        else:
            self.decoder_layer = TransformerEncoderLayer(d_model=256, nhead=8)
            self.transformer = TransformerEncoder(encoder_layer=self.decoder_layer, num_layers=6, residual=False)

        self.img_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=self._mask_size,
            sampling_ratio=2)

        if self.mask:
            self.mask_ratio = config.mask_ratio
            if config.mask_type == 'fixed':
                self.logger.debug(f"Masking with ratio {self.mask_ratio}")
            else:
                raise AttributeError("Masking with 15% is not implemented yet")
                # self.logger.debug("Masking randomly with 15% probability.")
        else:
            self.logger.debug("No masking..")

    def net_forward(self, base_features, objects):
        """Forward pass, override."""
        scales = self._box_scales
        # NOTE: I think you don't need to divide with scale and then rescale
        img_box = torch.FloatTensor(
            [[0, 0, self._img_shape[1] / scales[1],
              self._img_shape[0] / scales[0]]])
        img_box = img_box.type_as(base_features['0'])
        img_roi = self._rescale_boxes(img_box, self._box_scales)
        img_feats = self.img_roi_pool(base_features, [img_roi], [self._img_shape])
        img_feats = torch.flatten(img_feats, 2)

        rois = self._rescale_boxes(objects['boxes'], self._box_scales)
        obj_feats = self.orig_roi_pool(base_features, [rois], [self._img_shape])
        obj_feats = torch.mean(obj_feats, (2, 3))

        obj_pos = ObjPositionalEmbedding(rois, self._img_shape)

        return self._forward(
            img_feats, obj_feats, obj_pos, objects['mask_idx_obj'], objects
        )

    def _forward(self, img_feats, obj_feats, obj_pos, mask_idx_obj, objects):
        """Forward pass."""
        decoder_memory = img_feats.transpose(1, 2)
        decoder_memory = self.add_pos_emb(decoder_memory.transpose(0, 1))

        mask = torch.ones_like(obj_feats)
        num_masked_obj = obj_feats.size(0)

        if self.mask:
            masked_obj_ids = torch.zeros(obj_feats.size(0), dtype=torch.bool).to(self.device)
            num_masked_obj = int(self.mask_ratio * obj_feats.size(0))
            masked_obj_ids[:num_masked_obj] = True
            mask[:num_masked_obj] = torch.zeros_like(obj_feats[0])
            masked_input_no_pos = obj_feats * mask  # (num_obj, 256)
            masked_input_pos = torch.cat((masked_input_no_pos, obj_pos), dim=1)
            masked_input = self.obj_pos(masked_input_pos)
        else:
            masked_obj_ids = objects['mask_idx_obj']
            masked_input_no_pos = obj_feats * mask
            masked_input_pos = torch.cat((masked_input_no_pos, obj_pos), dim=1)
            masked_input = self.obj_pos(masked_input_pos)

        transformer_input = masked_input.unsqueeze(0).transpose(0, 1)

        if self.pretrain_arch == 'decoder':
            output, self_attn_weights, cross_attn_weights = self.transformer(transformer_input, decoder_memory)
        else:
            output, self_attn_weights, cross_attn_weights = self.transformer(transformer_input)
        output = output.transpose(0, 1)[0, :, :]  # (num_obj, 256)
        obj_scores = self.obj_classifier(output)

        if self.return_only_scores:
            return obj_scores, masked_obj_ids
        else:
            return obj_scores, masked_obj_ids, output, masked_obj_ids, self_attn_weights, cross_attn_weights


# from https://github.com/pytorch/examples/blob/main/word_language_model/model.py
class PositionalEmbedding(nn.Module):
    '''
    Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        pos_encoder = PositionalEncoding(d_model)
    '''

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Inputs of forward function
                Args:
                    x: the sequence fed to the positional encoder model (required).
                Shape:
                    x: [sequence length, batch size, embed dim]
                    output: [sequence length, batch size, embed dim]
                Examples:
                    output = pos_encoder(x)
        """
        # temp = self.pe[:x.size(0), :]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def ObjPositionalEmbedding(f_g, img_size, dim_g=256, wave_len=1000):
    x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=1)
    w, h = img_size

    delta_x = x_min / w
    delta_y = y_min / h
    delta_z = x_max / w
    delta_t = y_max / h
    size = delta_t.size()

    delta_x = delta_x.view(size[0], size[1], 1)
    delta_y = delta_y.view(size[0], size[1], 1)
    delta_z = delta_z.view(size[0], size[1], 1)
    delta_t = delta_t.view(size[0], size[1], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_z, delta_t), -1)

    feat_range = torch.arange(dim_g / 8).cuda()
    dim_mat = feat_range / (dim_g / 8)
    dim_mat = 1. / (torch.pow(wave_len, dim_mat))
    dim_mat = dim_mat.view(1, 1, -1)

    position_mat = position_mat.view(size[0], 4, -1)
    position_mat = 100. * position_mat

    mul_mat = position_mat * dim_mat
    mul_mat = mul_mat.view(size[0], -1)
    sin_mat = torch.sin(mul_mat)
    cos_mat = torch.cos(mul_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1)

    return embedding
