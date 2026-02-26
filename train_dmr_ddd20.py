import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
import argparse
import os
import numpy as np

# os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
import pandas as pd
from collections import OrderedDict
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from config_for_xhr import GlobalConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from data.dark_dataloader_ddd20_seq_len import CARLA_Data as EV
from pytorch_lightning import seed_everything
from torchvision import transforms
from torch.utils.data import ConcatDataset
import csv
torch.cuda.empty_cache()


def append_to_csv(step, loss_recon, loss_task, loss_repel, filename='episode_rewards.csv'):
    # 检查文件是否存在，如果不存在则创建并写入表头
    if not os.path.exists(filename):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Step', 'loss_recon', 'loss_task', 'loss_repel'])  # 写入表头

    # 以追加模式写入数据
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([step, loss_recon, loss_task, loss_repel])  # 写入数据


class Encoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, resnet_version=18, pretrained=False):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.obs_shape = obs_shape

        self.resnet = self._build_resnet(
            resnet_version=resnet_version,
            pretrained=pretrained,
            input_channels=obs_shape[0]
        )

        self.conv_shape = self._get_conv_shape(self.resnet)  # (1, C, H, W)
        conv_c, conv_h, conv_w = self.conv_shape[1:]

        self.fc = nn.Linear(conv_c * conv_h * conv_w, self.feature_dim)
        # out_shape = 8
        # self.fc = nn.Linear(512*out_shape*out_shape, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()

    def _get_conv_shape(self, resnet):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.obs_shape)  # e.g., (3, 256, 256)
            x = resnet.conv1(dummy_input)
            x = resnet.bn1(x)
            x = resnet.relu(x)
            x = resnet.maxpool(x)
            x = resnet.layer1(x)
            x = resnet.layer2(x)
            x = resnet.layer3(x)
            x = resnet.layer4(x)
            return x.shape  # (1, C, H, W)

    def _build_resnet(self, resnet_version, pretrained, input_channels):
        resnet_creator = getattr(models, f"resnet{resnet_version}")
        resnet = resnet_creator(pretrained=pretrained)

        if input_channels != 3:
            original_conv = resnet.conv1
            resnet.conv1 = nn.Conv2d(
                input_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias
            )
            # 初始化新卷积层
            nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
            # 移除原全连接层和池化层

        del resnet.fc
        del resnet.avgpool

        return resnet

    # def forward(self, obs, detach=False, vis=False):
    #     # print("obs.shape:", obs.shape)
    #     # 特征提取
    #     x = self.resnet.conv1(obs)
    #     x = self.resnet.bn1(x)
    #     x = self.resnet.relu(x)
    #     x = self.resnet.maxpool(x)
    #
    #     # 残差块处理
    #     x = self.resnet.layer1(x)
    #     x = self.resnet.layer2(x)
    #     x = self.resnet.layer3(x)
    #     conv = self.resnet.layer4(x)  # (512, 8, 8)
    #     if detach:
    #         conv = conv.detach()
    #
    #     h_fc = self.fc(conv.view(conv.size(0), -1))
    #     # out = self.ln(h_fc)
    #     out = nn.functional.normalize(h_fc)
    #
    #     self.outputs['h_fc'] = h_fc
    #     self.outputs['out'] = out
    #
    #     if vis:
    #         return out, conv
    #
    #     return out

    # huatu
    def forward(self, obs, detach=False, vis=False):
        # print("obs.shape:", obs.shape)
        # 特征提取
        x = self.resnet.conv1(obs)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        conv1 = x.clone()
        # 残差块处理
        x = self.resnet.layer1(x)
        conv2 = x.clone()

        x = self.resnet.layer2(x)
        conv3 = x.clone()

        x = self.resnet.layer3(x)
        conv4 = x.clone()

        x = self.resnet.layer4(x)  # (512, 8, 8)
        if detach:
            x = x.detach()

        h_fc = self.fc(x.view(x.size(0), -1))
        # out = self.ln(h_fc)
        out = nn.functional.normalize(h_fc)

        self.outputs['h_fc'] = h_fc
        self.outputs['out'] = out

        if vis:
            return out, [conv1, conv2, conv3, conv4]

        return out


class Decoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, conv_shape, target_hw=(256, 256), resnet_version=18):
        super().__init__()
        self.obs_shape = obs_shape  # (C, H, W)
        self.feature_dim = feature_dim
        self.target_hw = target_hw

        conv_c, conv_h, conv_w = conv_shape[1:]
        self.conv_c = conv_c
        self.conv_h = conv_h
        self.conv_w = conv_w
        self.fc = nn.Linear(feature_dim, conv_c * conv_h * conv_w)
        self.ln = nn.LayerNorm(conv_c * conv_h * conv_w)

        # # 初始全连接层（匹配Encoder最后的线性层）
        # self.fc = nn.Linear(feature_dim, 512*8*8)
        # self.ln = nn.LayerNorm(512*8*8)

        # 上采样模块配置（与ResNet对称）
        up_blocks = {
            18: [512, 256, 128, 64, 32],
            34: [512, 256, 128, 64, 32, 16],
            50: [2048, 1024, 512, 256, 128],
            101: [2048, 1024, 512, 256, 128, 64]
        }
        channels = up_blocks[resnet_version]

        # 上采样模块
        self.up_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.up_blocks.append(UpBlock(
                in_ch=channels[i],
                out_ch=channels[i + 1],
                use_residual=True if i < 2 else False  # 深层使用残差连接
            ))

        # 最终上采样到原始尺寸
        self.final_upsample = nn.Sequential(
            nn.Upsample(size=self.target_hw, mode='bilinear', align_corners=False),
            nn.Conv2d(channels[-1], channels[-1], 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 输出层
        self.final_conv = nn.Conv2d(channels[-1], obs_shape[0], 3, padding=1)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 初始投影 [B, D] -> [B, 512*8*8]
        x = self.ln(self.fc(x))
        # x = x.view(-1, 512, 8, 8)  # [B, 512, 8, 8]
        x = x.view(-1, self.conv_c, self.conv_h, self.conv_w)
        # 渐进上采样
        for block in self.up_blocks:
            x = block(x)

        # 最终上采样到原始尺寸
        x = self.final_upsample(x)  # [B, C_final, 256, 256]
        x = self.final_conv(x)  # [B, C_out, 256, 256]

        return torch.sigmoid(x)  # 假设输入图像归一化到[0,1]


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_residual=True):
        super().__init__()
        self.use_residual = use_residual

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        if use_residual:
            self.res_block = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch)
            )
            self.shortcut = nn.Conv2d(out_ch, out_ch, 1)

    def forward(self, x):
        x = self.upsample(x)

        if self.use_residual:
            residual = self.shortcut(x)
            x = self.res_block(x) + residual
            x = F.relu(x)

        return x


class TrajDecoder(nn.Module):
    def __init__(self, feature_dim, traj_shape):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(True),
            nn.Linear(feature_dim // 2, traj_shape)
        )

    def forward(self, com_z):
        traj = self.ffn(com_z)
        return traj


class DMR_TRAJ(nn.Module):
    def __init__(self, rgb_shape, dvs_shape, com_shape, latent_shape, traj_shape, temperature):
        super().__init__()
        aaa = 18
        # aaa = 34
        # aaa = 50
        # aaa = 101
        self.rgb_encoder = Encoder(rgb_shape, latent_shape, aaa)
        self.dvs_encoder = Encoder(dvs_shape, latent_shape, aaa)
        self.com_encoder = Encoder(com_shape, latent_shape, aaa)
        self.rgb_decoder = Decoder(rgb_shape, latent_shape, self.rgb_encoder.conv_shape, (256, 256), aaa)
        self.dvs_decoder = Decoder(dvs_shape, latent_shape, self.dvs_encoder.conv_shape, (256, 256), aaa)
        self.traj_decoder = TrajDecoder(latent_shape, traj_shape)

    def forward(self, rgb_input, dvs_input, vis=False):
        com_input = torch.cat([rgb_input, dvs_input], dim=1)

        if vis is True:
            rgb_h, rgb_conv = self.rgb_encoder(rgb_input, vis=vis)  # rgb noise
            dvs_h, dvs_conv = self.dvs_encoder(dvs_input, vis=vis)  # dvs noise
            com_z = self.com_encoder(com_input, vis=vis)  # com feature
            rgb_o = self.rgb_decoder(rgb_h + com_z)
            dvs_o = self.dvs_decoder(dvs_h + com_z)
            traj_o = self.traj_decoder(com_z)

            # return traj_o, rgb_o, dvs_o, [rgb_conv, dvs_conv, com_conv]
            return traj_o

        else:
            rgb_h = self.rgb_encoder(rgb_input, vis=vis)  # rgb noise
            dvs_h = self.dvs_encoder(dvs_input, vis=vis)  # dvs noise
            com_z = self.com_encoder(com_input, vis=vis)  # com feature
            rgb_o = self.rgb_decoder(rgb_h + com_z)
            dvs_o = self.dvs_decoder(dvs_h + com_z)
            traj_o = self.traj_decoder(com_z)

            return traj_o, rgb_o, dvs_o, [rgb_h, dvs_h, com_z]


class DMR_TRAJ_M1(nn.Module):
    def __init__(self, rgb_shape, dvs_shape, com_shape, latent_shape, traj_shape, temperature):
        super().__init__()
        # aaa = 18
        # aaa = 34
        aaa = 50
        # aaa = 101
        self.com_encoder = Encoder(com_shape, latent_shape, aaa)
        self.traj_decoder = TrajDecoder(latent_shape, traj_shape)

    def forward(self, rgb_input, dvs_input, vis=False):
        com_input = torch.cat([rgb_input, dvs_input], dim=1)
        if vis:
            com_z, com_conv = self.com_encoder(com_input, vis=vis)  # com feature
            traj_o = self.traj_decoder(com_z)

            return traj_o, None, None, [None, None, com_conv]

        else:
            com_z = self.com_encoder(com_input, vis=vis)  # com feature
            traj_o = self.traj_decoder(com_z)

            return traj_o, None, None, [None, None, com_z]


class DMR_TRAJ_M2(nn.Module):
    def __init__(self, rgb_shape, dvs_shape, com_shape, latent_shape, traj_shape, temperature):
        super().__init__()
        # aaa = 18
        # aaa = 34
        aaa = 50
        # aaa = 101
        self.rgb_encoder = Encoder(rgb_shape, latent_shape, aaa)
        self.dvs_encoder = Encoder(dvs_shape, latent_shape, aaa)
        self.traj_decoder = TrajDecoder(latent_shape * 2, traj_shape)

    def forward(self, rgb_input, dvs_input, vis=False):

        if vis:
            rgb_h, rgb_conv = self.rgb_encoder(rgb_input, vis=vis)  # rgb noise
            dvs_h, dvs_conv = self.dvs_encoder(dvs_input, vis=vis)  # dvs noise
            traj_o = self.traj_decoder(torch.cat([rgb_h, dvs_h], dim=-1))
            return traj_o, None, None, [rgb_conv, dvs_conv, None]

        else:
            rgb_h, rgb_conv = self.rgb_encoder(rgb_input, vis=vis)  # rgb noise
            dvs_h, dvs_conv = self.dvs_encoder(dvs_input, vis=vis)  # dvs noise
            traj_o = self.traj_decoder(torch.cat([rgb_h, dvs_h], dim=-1))

            return traj_o, None, None, [None, None, None]


# Set seed
seed = 42
seed_everything(seed)


class Dark_planner(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lr = config.lr
        self._load_weight()
        self.step = 0
        # 写死输入维度
        BBB = 5
        FFF = 3
        latent_shape = 512
        self.temperature = 1
        rgb_shape = (BBB, 3 * FFF, 256, 256)
        dvs_shape = (BBB, 3 * FFF, 256, 256)
        com_shape = (BBB, 2 * 3 * FFF, 256, 256)
        traj_shape = (BBB, 1)

        self.model = DMR_TRAJ(
            rgb_shape[1:], dvs_shape[1:], com_shape[1:],
            latent_shape, traj_shape[1], self.temperature
        )
        ############ M1
        # self.model = DMR_TRAJ_M1(
        #     rgb_shape[1:], dvs_shape[1:], com_shape[1:],
        #     latent_shape, traj_shape[1], self.temperature)

        ############ M2
        # self.model = DMR_TRAJ_M2(
        #     rgb_shape[1:], dvs_shape[1:], com_shape[1:],
        #     latent_shape, traj_shape[1], self.temperature)

    # 加载预训练权重
    def _load_weight(self):
        pass

    # 加载dict
    def _load_state_dict(self, il_net, rl_state_dict, key_word):
        rl_keys = [k for k in rl_state_dict.keys() if key_word in k]
        il_keys = il_net.state_dict().keys()
        assert len(rl_keys) == len(il_net.state_dict().keys()), f'mismatch number of layers loading {key_word}'
        new_state_dict = OrderedDict()
        for k_il, k_rl in zip(il_keys, rl_keys):
            new_state_dict[k_il] = rl_state_dict[k_rl]
        il_net.load_state_dict(new_state_dict)

    def forward(self, batch):
        pass

    # 单步训练，应该是重写了fit
    def training_step(self, data, batch_idx):
        dvs_image_frame = []
        aps_image_frame = []
        angle_frame = []
        for t in range(0, self.config.seq_len):  # 历史信息 + 当前信息
            dvs_image_frame.append(data['dvs_image_his_' + str(t)].to(dtype=torch.float32))
            aps_image_frame.append(data['aps_image_his_' + str(t)].to(dtype=torch.float32))

        gt_angle = data['gt_angle'].to(dtype=torch.float32)

        batch_size = dvs_image_frame[0].size(0)
        # his + cur
        rgb_input = torch.cat(aps_image_frame[1:4], dim=1)
        dvs_input = torch.cat(dvs_image_frame[1:4], dim=1)
        # print("---------------------------------------")
        # print("rgb_input:", rgb_input.shape)

        traj_o, rgb_o, dvs_o, [rgb_h, dvs_h, com_z] = self.model(rgb_input, dvs_input)

        # 定义loss
        # print("rgb_o:", rgb_o.shape)
        # print("rgb_input:", rgb_input.shape)

        loss_recon = F.mse_loss(rgb_o, rgb_input) + F.mse_loss(dvs_o, dvs_input)
        loss_task = F.mse_loss(traj_o.squeeze(), gt_angle)

        negative_keys = torch.cat([rgb_h, dvs_h], dim=0).clone()
        positive_logit = torch.sum(com_z * com_z, dim=1, keepdim=True)
        negative_logits = com_z @ (negative_keys.transpose(-2, -1))
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=com_z.device)
        loss_repel = F.cross_entropy(logits / self.temperature, labels, reduction='mean')

        # train_loss = loss_recon / loss_recon.detach() + \
        #              loss_task / loss_task.detach() + \
        #              loss_repel / loss_repel.detach()
        #
        train_loss = loss_recon + \
                     loss_task + \
                     loss_repel
        # train_loss = 0.1 * loss_task
        # train_loss = loss_task
        #
        filepath = os.path.join("/data2/xcm/SERF/log_xuhr/3", "{}.csv".format(seed))
        append_to_csv(self.step, loss_recon.item(), loss_task.item(), loss_repel.item(), filepath)
        self.step = self.step + 1
        # 日志
        self.log('train_loss', train_loss.item())
        return train_loss

    # 定义优化器
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=3e-5)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        return [optimizer], [lr_scheduler]

    # 单步验证，应该是重写了fit
    def validation_step(self, data, batch_idx):
        dvs_image_frame = []
        aps_image_frame = []
        for t in range(0, self.config.seq_len):  # 历史信息 + 当前信息
            dvs_image_frame.append(data['dvs_image_his_' + str(t)].to(dtype=torch.float32))
            aps_image_frame.append(data['aps_image_his_' + str(t)].to(dtype=torch.float32))

        gt_angle = data['gt_angle'].to(dtype=torch.float32)

        # his + cur
        rgb_input = torch.cat(aps_image_frame[1:4], dim=1)
        dvs_input = torch.cat(dvs_image_frame[1:4], dim=1)

        traj_o, rgb_o, dvs_o, [rgb_h, dvs_h, com_z] = self.model(rgb_input, dvs_input)

        # 定义loss
        loss_task = F.mse_loss(traj_o.squeeze(), gt_angle)
        val_loss = loss_task

        # 日志
        self.log('val_loss', val_loss, sync_dist=True)

        return val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--id', type=str,
                        default='xhr_hybrid_resnet_backbone_18_val',
                        help='Unique experiment identifier.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of train epochs.')
    #
    parser.add_argument('--root_dir', type=str, default='/data2/xcm/SERF/datasets/ddd20/export_data_hybrid', help='Path')
    parser.add_argument('--val_every', type=int, default=5, help='Validation frequency (epochs).')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--logdir', type=str, default='log_xuhr', help='Directory to log data to.')

    args = parser.parse_args()
    args.logdir = os.path.join(args.logdir, args.id)

    # Config
    config = GlobalConfig()

    # 定义数据增强操作
    event_dataset = EV(root=args.root_dir, csv_file="/train_carla.csv", config=config)
    total_size = int(len(event_dataset))

    val_size = int(total_size * 0.05)
    val_indices = np.random.choice(total_size, size=val_size, replace=False).tolist()

    del event_dataset

    train_dataset = EV(
        root=args.root_dir,
        csv_file="/train_hybrid.csv",
        config=config,
        is_val=False,  # 训练集
        val_indices=val_indices  # 传入验证集索引
    )

    # 验证集：使用验证集索引
    val_dataset = EV(
        root=args.root_dir,
        csv_file="/train_hybrid.csv",
        config=config,
        is_val=True,  # 验证集
        val_indices=val_indices  # 传入相同的验证集索引
    )
    print("train_dataset:", int(len(train_dataset)))
    print("val_dataset:", int(len(val_dataset)))


    # DataLoader
    dataloader_train = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    dataloader_val = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    # model
    model = Dark_planner(config)

    # 训练的回调函数
    checkpoint_callback = ModelCheckpoint(save_weights_only=False, mode="min", monitor="val_loss", save_top_k=4,
                                          save_last=True,
                                          dirpath=args.logdir, filename="best_{epoch:02d}-{val_loss:.3f}")
    checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"
    trainer = pl.Trainer.from_argparse_args(args,
                                            default_root_dir=args.logdir,
                                            # 1,2,3,4,5,6,7,8,9,10,11,12,13
                                            # gpus=[0,1,2,3,4,5,6,7,8,9,10,11,12],
                                            gpus=[4],
                                            accelerator='ddp',
                                            sync_batchnorm=True,
                                            plugins=DDPPlugin(find_unused_parameters=True),
                                            profiler='simple',
                                            benchmark=True,
                                            log_every_n_steps=1,
                                            flush_logs_every_n_steps=5,
                                            callbacks=[checkpoint_callback
                                                       ],
                                            check_val_every_n_epoch=args.val_every,
                                            max_epochs=args.epochs,
                                            # resume_from_checkpoint="/home/cg/DAFuser/log/DAFuser_without_action_4_no-share-MHSArp-and-MHSAep_fusion=3_frame=4_normal/epoch=119-last.ckpt"
                                            )

    trainer.fit(model, dataloader_train, dataloader_val)
