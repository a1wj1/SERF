import argparse
import os
# os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
import pandas as pd
from collections import OrderedDict
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from config import GlobalConfig
from models.SERF_model import SERF
from data.dataloader_carla import CARLA_Data as EV
from pytorch_lightning import seed_everything
from torchvision import transforms
from torch.utils.data import ConcatDataset

torch.cuda.empty_cache()

# Set seed
seed = 42
seed_everything(seed)


class Dark_planner(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.lr = config.lr
        self._load_weight()
        self.model = SERF(config)

    # 加载预训练权重
    def _load_weight(self):
        pass

    # 加载dict
    def _load_state_dict(self, il_net, rl_state_dict, key_word):
        pass

    def forward(self, batch):
        pass

    # 单步训练，应该是重写了fit
    def training_step(self, data, batch_idx):
        dvs_image_frame = []
        aps_image_frame = []
        for t in range(0, self.config.seq_len):  # 历史信息 + 当前信息
            dvs_image_frame.append(data['dvs_image_his_' + str(t)].to(dtype=torch.float32))
            aps_image_frame.append(data['aps_image_his_' + str(t)].to(dtype=torch.float32))

        gt_angle = data['gt_angle'].to(dtype=torch.float32)

        pred_a = self.model(dvs_image_frame, aps_image_frame)
        train_loss = F.l1_loss(pred_a.squeeze(1), gt_angle, reduction='none').mean()

        # 日志
        self.log('train_loss', train_loss.item())
        return train_loss

    # 定义优化器
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
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
        pred_a = self.model(dvs_image_frame, aps_image_frame)
        val_loss = float(F.l1_loss(pred_a.squeeze(1), gt_angle, reduction='none').mean())

        # 日志
        self.log('val_loss', val_loss, sync_dist=True)
        return val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--id', type=str, default='SERF_carla', help='Unique experiment identifier.')
    parser.add_argument('--epochs', type=int, default=150, help='Number of train epochs.')
    #
    parser.add_argument('--root_dir', type=str, default='/home/xxx/steer_test/carla/data', help='Path')
    parser.add_argument('--val_every', type=int, default=5, help='Validation frequency (epochs).')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')

    args = parser.parse_args()
    args.logdir = os.path.join(args.logdir, args.id)

    # Config
    config = GlobalConfig()

    # 定义数据增强操作
    event_dataset = EV(root=args.root_dir, csv_file="/test_carla.csv", config=config)
    dataset_size = int(len(event_dataset))
    del event_dataset
    split_point = int(dataset_size * 0.8)

    train_dataset = EV(root=args.root_dir, csv_file="/test_carla.csv", config=config, select_range=(0,split_point))
    test_dataset = EV(root=args.root_dir, csv_file="/test_carla.csv", config=config, select_range=(split_point,dataset_size))


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
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    # model
    model = Dark_planner(config)

    # 训练的回调函数
    checkpoint_callback = ModelCheckpoint(save_weights_only=False, mode="min", monitor="val_loss", save_top_k=2,
                                          save_last=True,
                                          dirpath=args.logdir, filename="best_{epoch:02d}-{val_loss:.3f}")
    checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"
    trainer = pl.Trainer.from_argparse_args(args,
                                            default_root_dir=args.logdir,
                                            # 1,2,3,4,5,6,7,8,9,10,11,12,13
                                            # gpus=[0,1,2,3,4,5,6,7,8,9,10,11,12],
                                            gpus=[1, 2],  # 我们的方法必须用两个gpu跑
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
                                            # resume_from_checkpoint=".ckpt"
                                            )

    trainer.fit(model, dataloader_train, dataloader_val)
