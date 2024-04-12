# -*- coding: utf-8 -*-
"""Masked Bounding Box Reconstruction Pre-trained Network, by Anastasakis et al., 2024"""

import math
from numpy import pi
import torch
from torch import nn
from torchvision.ops import MultiScaleRoIAlign
from common.models.transformers.base_transformer import TransformerDecoder, TransformerDecoderLayer
from common.models.transformers.base_transformer import TransformerEncoder, TransformerEncoderLayer

from .base_sg_generator import BaseSGGenerator


class MBBR(BaseSGGenerator):
    """Extends PyTorch nn.Module."""

    def __init__(self, config, mask=True):
        """Initialize layers."""
        super().__init__(config, {'base_features', 'pool_features'})

        self.logger = config.logger
        self.device = config.device
        self.mask_type = config.mask_type
        self.projection_head = config.projection_head

        # Language features
        self.fc_object_lang = nn.Sequential(nn.Linear(300, 256), nn.ReLU())
        self.fc_object_feats = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())

        self.img_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=self._mask_size,
            sampling_ratio=2)

        self.img_spat_enc = self.get_spatial_encodings(self._mask_size)

        self.add_pos_emb = PositionalEmbedding(d_model=256)

        self.use_pos_enc = config.use_pos_enc
        if config.use_pos_enc:
            if config.use_pos_enc == 'learnable':
                self.obj_pos = nn.Linear(256 * 3, 256)  # with learnable positional embeds
                raise NotImplementedError("Not implemented yet learnable pos emb.")
            else:
                self.obj_pos = nn.Linear(512, 256)
        else:
            self.obj_pos = nn.Linear(512, 256)

        assert config.pretrain_arch == 'encoder' or config.pretrain_arch == 'decoder', "Please configure a valid transformer architecture: 'encoder' or 'decoder'"
        self.pretrain_arch = config.pretrain_arch
        if self.pretrain_arch == 'decoder':
            self.logger.debug("Using Decoder..")
            self.decoder_layer = TransformerDecoderLayer(d_model=256, nhead=8)
            self.decoder = TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=6, residual=False,
                                              pre_last_layer=config.pre_last_layer)
        else:
            self.logger.debug("Using Encoder..")
            self.decoder_layer = TransformerEncoderLayer(d_model=256, nhead=8)
            self.decoder = TransformerEncoder(encoder_layer=self.decoder_layer, num_layers=6, residual=False)

        if self.projection_head:
            self.proj_head = nn.Linear(256, 256)

        self.mask = mask if not config.mask_before_frcnn else False

        if self.use_pos_enc:
            self.logger.debug(f"Using {self.use_pos_enc} extra positional encoding.")

        if self.mask:
            self.mask_ratio = config.mask_ratio
            if self.mask_type == 'fixed':
                self.logger.debug(f"Masking with ratio {self.mask_ratio}")
            else:
                self.logger.debug("Masking randomly with 15% probability.")
        else:
            self.logger.debug("No masking..")

    # prepare features (arguments) for _forward
    def net_forward(self, base_features, objects, pairs):
        """Forward pass, override."""
        # Get image features
        scales = self._box_scales
        img_box = torch.FloatTensor(
            [[0, 0, self._img_shape[1] / scales[1],
              self._img_shape[0] / scales[0]]])
        img_box = img_box.type_as(base_features['0'])
        rois = self._rescale_boxes(img_box, self._box_scales)
        img_feats = self.img_roi_pool(base_features, [rois], [self._img_shape])

        rois = self._rescale_boxes(objects['boxes'], self._box_scales)
        obj_feats = self.orig_roi_pool(base_features, [rois], [self._img_shape])
        obj_feats = torch.mean(obj_feats, (2, 3))

        img_feats = torch.flatten(img_feats, 2)

        obj_pos = ObjPositionalEmbedding(rois, self._img_shape)

        return self._forward(
            self.get_pred_pooled_features(
                base_features,
                objects['boxes'][pairs[:, 0]], objects['boxes'][pairs[:, 1]]
            ),
            img_feats,
            obj_feats,
            objects,
            obj_pos
        )

    def _forward(self, pred_feats, img_feats, all_obj_feats, objects, obj_pos):
        """Forward pass, returns output scores."""

        decoder_memory = img_feats.transpose(1, 2)
        decoder_memory = self.add_pos_emb(decoder_memory.transpose(0, 1))

        mask = torch.ones_like(all_obj_feats)
        if self.mask:
            if self.mask_type == 'fixed':
                masked_obj_ids = torch.zeros(all_obj_feats.size(0), dtype=torch.bool).to(self.device)
                num_masked_obj = int(self.mask_ratio * all_obj_feats.size(0))
                masked_obj_ids[:num_masked_obj] = True
                mask[:num_masked_obj] = torch.zeros_like(all_obj_feats[0])
                masked_input_no_pos = all_obj_feats * mask  # (num_obj, 256)
            else:
                # another masking method: mask all objects with 15% probability
                num_objs = all_obj_feats.size(0)
                masked_obj_ids = torch.rand(num_objs)
                masked_obj_ids = masked_obj_ids < 0.15
                mask[masked_obj_ids] = 0
                masked_input_no_pos = all_obj_feats * mask

            if self.use_pos_enc == 'fixed':
                masked_input_no_pos = masked_input_no_pos.unsqueeze(0).transpose(0, 1)
                masked_input_no_pos = self.add_pos_emb(masked_input_no_pos)
                masked_input_no_pos = masked_input_no_pos.transpose(0, 1)[0, :, :]
            masked_input_pos = torch.cat((masked_input_no_pos, obj_pos), dim=1)
            masked_input = self.obj_pos(masked_input_pos)
        else:
            masked_obj_ids = objects['mask_idx_obj']
            masked_input_no_pos = all_obj_feats * mask
            if self.use_pos_enc == 'fixed':
                masked_input_no_pos = masked_input_no_pos.unsqueeze(0).transpose(0, 1)
                masked_input_no_pos = self.add_pos_emb(masked_input_no_pos)
                masked_input_no_pos = masked_input_no_pos.transpose(0, 1)[0, :, :]
            masked_input_pos = torch.cat((masked_input_no_pos, obj_pos), dim=1)
            masked_input = self.obj_pos(masked_input_pos)

        decoder_input = masked_input.unsqueeze(0).transpose(0, 1)
        if self.pretrain_arch == 'decoder':
            # original implementation
            output, self_attn_weights, cross_attn_weights = self.decoder(decoder_input, decoder_memory)
        else:
            output, self_attn_weights, cross_attn_weights = self.decoder(decoder_input)

        output = output.transpose(0, 1)[0, :, :]  # (num_obj, 256)
        og_output = output
        if self.projection_head:
            output = self.proj_head(output)

        return output, all_obj_feats \
            , og_output, masked_obj_ids, self_attn_weights, cross_attn_weights, pred_feats

    @staticmethod
    def get_spatial_encodings(n):
        # try adding logarithmic scaling of frequencies
        x_ = torch.arange(n).view(1, -1).expand(n, -1) / n
        y_ = torch.arange(n).view(-1, 1).expand(-1, n) / n
        spat_enc = torch.stack((x_, y_), dim=0)
        pos_enc = torch.cat([
            torch.cat((torch.sin(k * 2 * pi * spat_enc),
                       torch.cos(k * 2 * pi * spat_enc)), dim=0)
            for k in range(1, n // 2)
        ], dim=0)
        pos_enc = torch.cat((pos_enc, spat_enc), dim=0)
        return pos_enc.unsqueeze(0)


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
