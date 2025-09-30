import torch
import torchvision.models as models
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from functools import partial
from itertools import repeat
import collections.abc
import math
import torch.nn.functional as F
from models.pos_embed import get_2d_sincos_pos_embed
from torch.distributions import normal
from models.former import Former, Res2Former
import sys


# =============================================================================
# Processing blocks
# X-attention input
#   Q/z_input         -> (#latent_embs, batch_size, embed_dim)
#   K/V/x             -> (#events, batch_size, embed_dim)
#   key_padding_mask  -> (batch_size, #event)
# output -> (#latent_embs, batch_size, embed_dim)
# =============================================================================
class AttentionBlock(nn.Module):  # PerceiverAttentionBlock
    def __init__(self, opt_dim, heads, dropout, att_dropout, **args):
        super(AttentionBlock, self).__init__()

        self.layer_norm_x = nn.LayerNorm([opt_dim])
        self.layer_norm_1 = nn.LayerNorm([opt_dim])
        self.layer_norm_att = nn.LayerNorm([opt_dim])

        self.attention = nn.MultiheadAttention(
            opt_dim,  # embed_dim
            heads,  # num_heads
            dropout=att_dropout,
            bias=True,
            add_bias_kv=True,
            sigma=3.0,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(opt_dim, opt_dim)
        self.layer_norm_2 = nn.LayerNorm([opt_dim])
        self.linear2 = nn.Linear(opt_dim, opt_dim)
        self.linear3 = nn.Linear(opt_dim, opt_dim)

    def forward(self, x, z_input, mask=None, q_mask=None, rank=1, **args):
        x = self.layer_norm_x(x)
        z = self.layer_norm_1(z_input)

        z_att, _ = self.attention(rank, z, x, x, key_padding_mask=mask, attn_mask=q_mask)  # Q, K, V

        z_att = z_att + z_input
        z = self.layer_norm_att(z_att)

        z = self.dropout(z)
        z = self.linear1(z)
        z = torch.nn.GELU()(z)

        z = self.layer_norm_2(z)
        z = self.linear2(z)
        z = torch.nn.GELU()(z)
        z = self.dropout(z)
        z = self.linear3(z)

        return z + z_att


class TransformerBlock(nn.Module):
    def __init__(self, opt_dim, latent_blocks, dropout, att_dropout, heads, **args):
        super(TransformerBlock, self).__init__()

        self.latent_attentions = nn.ModuleList([
            AttentionBlock(opt_dim=opt_dim, heads=heads, dropout=dropout, att_dropout=att_dropout) for _ in
            range(latent_blocks)
        ])

    def forward(self, x_input, z, mask=None, q_mask=None, rank=1, **args):
        # self-attention
        for latent_attention in self.latent_attentions:
            z = latent_attention(x_input, z, q_mask=q_mask, rank=rank)
        return z


class CrossAttentionBlock(nn.Module):
    def __init__(self, opt_dim, dropout, att_dropout, cross_heads, **args):
        super(CrossAttentionBlock, self).__init__()

        self.cross_attention = AttentionBlock(opt_dim=opt_dim, heads=cross_heads, dropout=dropout,
                                              att_dropout=att_dropout)

    def forward(self, x_input, z, mask=None, q_mask=None, rank=1, **args):
        # cross-attention
        z = self.cross_attention(x_input, z, mask=mask, q_mask=q_mask, rank=rank)
        return z


class SERF(nn.Module):
    def __init__(self, config):
        super(SERF, self).__init__()

        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.in_chans = config.in_chans
        self.embed_dim = config.embed_dim
        self.att_heads = config.att_heads
        self.cross_heads = config.cross_heads
        self.latent_blocks = config.latent_blocks
        self.num_latent_vectors = config.num_latent_vectors
        self.dropout = config.dropout
        self.att_dropout = config.att_dropout
        self.fusion_time = config.fusion_time

        self.patch_embed_rgb = PatchEmbed(self.img_size, self.patch_size, self.in_chans, self.embed_dim)
        self.patch_embed_dvs = PatchEmbed(self.img_size, self.patch_size, self.in_chans, self.embed_dim)
        num_patches = self.patch_embed_rgb.num_patches
        self.num_patches = num_patches

        self.pos_embed_rgb = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim),
                                          requires_grad=False)  # fixed sin-cos embedding

        self.pos_embed_dvs = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim),
                                          requires_grad=False)  # fixed sin-cos embedding

        self.cross_attention = CrossAttentionBlock(opt_dim=self.embed_dim,
                                                   latent_blocks=self.latent_blocks,
                                                   dropout=self.dropout,
                                                   att_dropout=self.att_dropout,
                                                   cross_heads=self.cross_heads)

        self.cross_attention_latent = CrossAttentionBlock(opt_dim=self.embed_dim,
                                                          latent_blocks=self.latent_blocks,
                                                          dropout=self.dropout,
                                                          att_dropout=self.att_dropout,
                                                          cross_heads=self.cross_heads)

        self.latent_attention = TransformerBlock(opt_dim=self.embed_dim,
                                                 latent_blocks=self.latent_blocks,
                                                 dropout=self.dropout,
                                                 att_dropout=self.att_dropout,
                                                 heads=self.att_heads)

        self.self_attention = TransformerBlock(opt_dim=self.embed_dim,
                                               latent_blocks=self.latent_blocks,
                                               dropout=self.dropout,
                                               att_dropout=self.att_dropout,
                                               heads=self.att_heads)

        self.self_attention_head = TransformerBlock(opt_dim=self.embed_dim,
                                                    latent_blocks=self.latent_blocks,
                                                    dropout=self.dropout,
                                                    att_dropout=self.att_dropout,
                                                    heads=self.att_heads)
        #
        # 定义8x8 tensor，全部设为True
        tensor = torch.ones((8, 8), dtype=torch.bool)
        # 定义窗口宽度和高度
        window_width = 3
        window_height = 3
        offset_x = window_height // 2
        offset_y = window_width // 2
        # 定义存放所有情况的列表
        tensors = []
        # 遍历所有位置
        for i in range(8):
            for j in range(8):
                # 复制原始张量
                temp = tensor.clone()
                # 为每一种情况创造一个窗口覆盖区域，设为False
                for dx in range(-offset_x, offset_x + 1):
                    for dy in range(-offset_y, offset_y + 1):
                        x, y = i + dx, j + dy
                        # 只处理有效区域
                        if 0 <= x < 8 and 0 <= y < 8:
                            temp[x, y] = False
                # 添加到列表中
                tensors.append(temp.flatten())
        self.mask = torch.stack(tensors, dim=1)
        self.rank = 1
        #
        self.global_token = nn.Parameter(
            normal.Normal(0.0, 0.2).sample((self.num_latent_vectors, self.embed_dim)).clip(-2, 2),
            requires_grad=True)
        self.former = Former(dim=self.embed_dim, depth=2)
        # decoder
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.decoder_norm = norm_layer(self.embed_dim)
        self.decoder = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, 1))
        #
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed_rgb = get_2d_sincos_pos_embed(self.pos_embed_rgb.shape[-1], int(self.num_patches ** .5),
                                                cls_token=False)
        self.pos_embed_rgb.data.copy_(torch.from_numpy(pos_embed_rgb).float().unsqueeze(0))
        pos_embed_dvs = get_2d_sincos_pos_embed(self.pos_embed_dvs.shape[-1], int(self.num_patches ** .5),
                                                cls_token=False)
        self.pos_embed_dvs.data.copy_(torch.from_numpy(pos_embed_dvs).float().unsqueeze(0))

        # share加这个效果好像好一点
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed_rgb.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.patch_embed_dvs.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, dvs_image_frame, aps_image_frame):
        batch_size = dvs_image_frame[0].size(0)

        # his + cur
        inputs_rgb = torch.stack(aps_image_frame[4:8], dim=1)
        x_rgb = inputs_rgb.contiguous().view((-1,) + inputs_rgb.shape[2:])

        inputs_event_1 = torch.stack(dvs_image_frame[4:8], dim=1)
        x_event_1 = inputs_event_1.contiguous().view((-1,) + inputs_event_1.shape[2:])

        inputs_event_2 = torch.stack(dvs_image_frame[0:4], dim=1)
        x_event_2 = inputs_event_2.contiguous().view((-1,) + inputs_event_2.shape[2:])

        # Initial latent vectors
        latent_vectors = self.global_token.unsqueeze(1)
        latent_vectors = latent_vectors.expand(-1, x_rgb.size(0), -1)  # (num_latent_vectors, batch_size, embed_dim)

        #################################rgb与event的交互模块#####################################
        x = self.patch_embed_rgb(x_rgb)
        rgb = x + self.pos_embed_rgb
        rgb = rgb.permute(1, 0, 2)  # (tokens, batch_size, emb_dim)
        rgb = self.self_attention_head(rgb, rgb)
        inp_q_rgb = rgb
        inp_kv_rgb = rgb

        for index_event in range(0, 2):
            x_event = None
            if index_event == 0:
                x_event = x_event_1
                self.rank = 1
            elif index_event == 1:
                x_event = x_event_2
                self.rank = 1

            event = x_event
            x = self.patch_embed_dvs(event)
            event = x + self.pos_embed_dvs
            event = event.permute(1, 0, 2)  # (tokens, batch_size, emb_dim)
            event = self.self_attention_head(event, event)

            for t in range(3):
                self.mask = self.mask.to(inp_q_rgb.device)
                cross_emd_q = self.cross_attention(x_input=event, z=inp_q_rgb, mask=None, q_mask=self.mask,
                                                   rank=self.rank)
                inp_q_rgb = self.self_attention(cross_emd_q, cross_emd_q)

                cross_emd_kv = self.cross_attention(x_input=inp_kv_rgb, z=event, mask=None, q_mask=None, rank=self.rank)
                inp_kv_rgb = self.self_attention(cross_emd_kv, cross_emd_kv)
                latent_vectors = self.cross_attention_latent(cross_emd_q + cross_emd_kv,
                                                             latent_vectors) + latent_vectors

        fusion = self.latent_attention(inp_kv_rgb, inp_q_rgb) + self.latent_attention(inp_q_rgb, inp_kv_rgb)

        latent_vectors = latent_vectors.permute(1, 0, 2)  # (batch_size, latent_embs, emb_dim)
        ###############################Fusion#######################################
        emd = self.former(latent_vectors) + fusion.permute(1, 0, 2)

        ###############################Decoder#######################################
        decoder = self.decoder_norm(emd)
        # predictor projection
        fuse = decoder.mean(dim=1)
        fuse = fuse.contiguous().view(batch_size, -1)
        fuse = self.decoder(fuse)

        return fuse
