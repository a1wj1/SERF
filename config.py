import os


class GlobalConfig:
    img_size = 256  # hybrid和action_spatial: 128，其他：256
    patch_size = 32
    fusion_time = 1

    in_chans = 3  # hybrid和action_spatial: 64，其他：3
    embed_dim = 512  # steer: 512
    resnet = 50
    num_latent_vectors = 64  # 64

    dropout = 0.3
    cross_heads = 4
    att_dropout = 0.3
    att_heads = 4
    latent_blocks = 2

    # his action
    action_n_head = 4
    action_block_exp = 4
    action_attn_pdrop = 0.1
    action_resid_pdrop = 0.1
    action_n_layer = 2

    lr = 0.0001

    seq_len = 8

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
