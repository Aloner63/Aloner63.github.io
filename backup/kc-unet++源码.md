train.py
```
import argparse
import os
from collections import OrderedDict
from glob import glob

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
import albumentations as A
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

import archs
import losses
from dataset import Dataset
from metrics import iou_score, dice_coef, precision_score, recall_score
from utils import AverageMeter, str2bool

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=150, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UKAN_NestedUNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                             ' | '.join(ARCH_NAMES) +
                             ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=512, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=512, type=int,
                        help='image height')

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                             ' | '.join(LOSS_NAMES) +
                             ' (default: BCEDiceLoss)')

    # dataset
    parser.add_argument('--dataset', default='dsb2018_96_1000',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.jpg',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')

    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'precision': AverageMeter(),
                  'recall': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            output = outputs[-1]
        else:
            output = model(input)
            loss = criterion(output, target)

        # compute metrics
        iou = iou_score(output, target)
        dice = dice_coef(output, target)
        precision = precision_score(output, target)
        recall = recall_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))
        avg_meters['precision'].update(precision, input.size(0))
        avg_meters['recall'].update(recall, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice', avg_meters['dice'].avg),
            ('precision', avg_meters['precision'].avg),
            ('recall', avg_meters['recall'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('precision', avg_meters['precision'].avg),
                        ('recall', avg_meters['recall'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'precision': AverageMeter(),
                  'recall': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                output = outputs[-1]
            else:
                output = model(input)
                loss = criterion(output, target)

            # compute metrics
            iou = iou_score(output, target)
            dice = dice_coef(output, target)
            precision = precision_score(output, target)
            recall = recall_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['precision'].update(precision, input.size(0))
            avg_meters['recall'].update(recall, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('precision', avg_meters['precision'].avg),
                ('recall', avg_meters['recall'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('precision', avg_meters['precision'].avg),
                        ('recall', avg_meters['recall'].avg)])


def main():
    config = vars(parse_args())

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    # Enable multi-GPU support with DataParallel
    if torch.cuda.device_count() > 1:
        print(f"=> Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    train_transform = Compose([
        A.RandomRotate90(),
        A.HorizontalFlip(p=0.5),  # 50% 概率水平翻转
        A.VerticalFlip(p=0.5),  # 50% 概率垂直翻转
        OneOf([
            A.HueSaturationValue(),
            A.RandomBrightnessContrast(),
        ], p=1),
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])

    val_transform = Compose([
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    # 创建CSV文件
    df = pd.DataFrame(columns=['epoch', 'loss', 'iou', 'dice', 'precision', 'recall',
                              'val_loss', 'val_iou', 'val_dice', 'val_precision', 'val_recall'])
    df.to_csv('models/%s/log.csv' % config['name'], index=False)

    best_iou = 0
    best_epoch = 0
    for epoch in range(config['epochs']):
        print('\nEpoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - dice %.4f - precision %.4f - recall %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f - val_precision %.4f - val_recall %.4f'
              % (train_log['loss'], train_log['iou'], train_log['dice'], train_log['precision'], train_log['recall'],
                 val_log['loss'], val_log['iou'], val_log['dice'], val_log['precision'], val_log['recall']))

        df = pd.DataFrame([[epoch, train_log['loss'], train_log['iou'], train_log['dice'], train_log['precision'], train_log['recall'],
                          val_log['loss'], val_log['iou'], val_log['dice'], val_log['precision'], val_log['recall']]],
                         columns=['epoch', 'loss', 'iou', 'dice', 'precision', 'recall',
                                 'val_loss', 'val_iou', 'val_dice', 'val_precision', 'val_recall'])
        df.to_csv('models/%s/log.csv' % config['name'], mode='a', header=False, index=False)

        if val_log['iou'] > best_iou:
            print("=> saved best model")
            best_iou = val_log['iou']
            best_epoch = epoch
            torch.save(model.state_dict(), 'models/%s/model.pth' % config['name'])

    print("=> Best IoU: %.4f at epoch %d" % (best_iou, best_epoch))


if __name__ == '__main__':
    main()
```

archs.py
```

import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from kan import KANLinear
from model.attention.CBAM import CBAMBlock  # 导入CBAM模块

__all__ = ['UKAN_NestedUNet']

class KANLayer(nn.Module):
    """
    KAN层实现，基于Kolmogorov-Arnold网络
    这是一个特殊的神经网络层，使用样条函数来增强网络的表达能力
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # 网格大小，控制样条函数的复杂度
        grid_size = 3
        # 样条函数的阶数
        spline_order = 2
        # 噪声缩放因子，用于初始化
        scale_noise = 0.1
        # 基础缩放因子
        scale_base = 1.0
        # 样条缩放因子
        scale_spline = 1.0
        # 基础激活函数
        base_activation = nn.SiLU
        # 网格epsilon值，防止网格点过于接近
        grid_eps = 0.02
        # 网格范围
        grid_range = [-1, 1]

        # 第一个KAN线性层，将输入特征映射到隐藏特征
        self.fc1 = KANLinear(
            in_features,
            hidden_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )
        # 第二个KAN线性层，将隐藏特征映射到输出特征
        self.fc2 = KANLinear(
            hidden_features,
            out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )
        # 深度可分离卷积，用于捕获空间信息
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        # 批归一化层
        self.bn = nn.BatchNorm2d(hidden_features)
        # ReLU激活函数
        self.relu = nn.ReLU()
        # Dropout层，用于正则化
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        """
        前向传播函数
        x: 输入特征 [B, N, C]
        H, W: 特征图的高和宽
        """
        B, N, C = x.shape
        # 应用第一个KAN线性层
        x = self.fc1(x.reshape(B * N, C)).reshape(B, N, -1)
        # 重塑为图像格式以应用卷积
        x = x.transpose(1, 2).view(B, -1, H, W)
        # 应用深度可分离卷积、批归一化和ReLU
        x = self.relu(self.bn(self.dwconv(x)))
        # 重塑回序列格式
        x = x.flatten(2).transpose(1, 2)
        # 应用第二个KAN线性层
        x = self.fc2(x.reshape(B * N, -1)).reshape(B, N, -1)
        # 应用dropout并返回
        return self.drop(x)

class KANBlock(nn.Module):
    """
    KAN块，包含一个LayerNorm和一个KANLayer，并使用残差连接
    """
    def __init__(self, dim, drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        # DropPath用于随机丢弃残差连接，增强正则化
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 层归一化
        self.norm = norm_layer(dim)
        # KAN层
        self.layer = KANLayer(in_features=dim, hidden_features=dim, drop=drop)

    def forward(self, x, H, W):
        """
        前向传播函数，实现残差连接
        """
        return x + self.drop_path(self.layer(self.norm(x), H, W))

class PatchEmbed(nn.Module):
    """
    图像块嵌入层，将图像转换为序列表示
    """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # 计算输出特征图的高和宽
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        # 计算图像块的数量
        self.num_patches = self.H * self.W
        # 卷积层，用于提取图像块特征
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        # 层归一化
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        前向传播函数
        将图像转换为序列表示
        """
        # 应用卷积
        x = self.proj(x)
        # 获取输出特征图的尺寸
        _, _, H, W = x.shape
        # 将特征图展平为序列
        x = x.flatten(2).transpose(1, 2)
        # 应用层归一化
        x = self.norm(x)
        return x, H, W

class VGGBlock(nn.Module):
    """
    VGG块，包含两个卷积层，每个卷积层后跟批归一化和ReLU激活
    """
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        # ReLU激活函数
        self.relu = nn.ReLU(inplace=True)
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        # 第一个批归一化层
        self.bn1 = nn.BatchNorm2d(middle_channels)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        # 第二个批归一化层
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        前向传播函数
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

class UKAN_NestedUNet(nn.Module):
    """
    UKAN嵌套UNet模型，将KAN模块集成到UNet++架构中
    KAN模块主要在深层特征(x2_0, x2_2, x3_0, x4_0, x3_1)中发挥作用
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, **kwargs):
        super().__init__()
        # 定义每层的滤波器数量
        self.nb_filter = [32, 64, 128, 256, 512]
        # 是否使用深度监督
        self.deep_supervision = deep_supervision
        # 最大池化层，用于下采样
        self.pool = nn.MaxPool2d(2, 2)
        # 上采样层，用于上采样
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 编码器部分
        # 第一层编码器
        self.conv0_0 = VGGBlock(input_channels, self.nb_filter[0], self.nb_filter[0])
        # 第二层编码器
        self.conv1_0 = VGGBlock(self.nb_filter[0], self.nb_filter[1], self.nb_filter[1])
        # 第三层编码器
        self.conv2_0 = VGGBlock(self.nb_filter[1], self.nb_filter[2], self.nb_filter[2])
        
        # 新增：为x2_0添加PatchEmbed和KANBlock
        self.patch_embed2 = PatchEmbed(img_size=img_size//4, patch_size=3, stride=1, in_chans=self.nb_filter[2], embed_dim=self.nb_filter[2])
        self.block2 = KANBlock(dim=self.nb_filter[2], norm_layer=nn.LayerNorm)
        
        # 第四层编码器
        self.conv3_0 = VGGBlock(self.nb_filter[2], self.nb_filter[3], self.nb_filter[3])
        # 第四层的图像块嵌入，将特征转换为序列表示以应用KAN
        self.patch_embed3 = PatchEmbed(img_size=img_size//8, patch_size=3, stride=2, in_chans=self.nb_filter[3], embed_dim=self.nb_filter[3])
        # 第四层的KAN块，增强特征表达能力
        self.block3 = KANBlock(dim=self.nb_filter[3], norm_layer=nn.LayerNorm)
        # 第五层的图像块嵌入
        self.patch_embed4 = PatchEmbed(img_size=img_size//16, patch_size=3, stride=2, in_chans=self.nb_filter[3], embed_dim=self.nb_filter[4])
        # 第五层的KAN块
        self.block4 = KANBlock(dim=self.nb_filter[4], norm_layer=nn.LayerNorm)

        # 解码器部分
        # 第一层解码器
        self.conv0_1 = VGGBlock(self.nb_filter[0]+self.nb_filter[1], self.nb_filter[0], self.nb_filter[0])
        # 第二层解码器
        self.conv1_1 = VGGBlock(self.nb_filter[1]+self.nb_filter[2], self.nb_filter[1], self.nb_filter[1])
        # 第三层解码器
        self.conv2_1 = VGGBlock(self.nb_filter[2]+self.nb_filter[3], self.nb_filter[2], self.nb_filter[2])
        # 第四层解码器
        self.conv3_1 = VGGBlock(self.nb_filter[3]+self.nb_filter[4], self.nb_filter[3], self.nb_filter[3])
        # 第四层解码器的KAN块
        self.dblock3 = KANBlock(dim=self.nb_filter[3], norm_layer=nn.LayerNorm)
        # 嵌套连接的解码器
        self.conv0_2 = VGGBlock(self.nb_filter[0]*2+self.nb_filter[1], self.nb_filter[0], self.nb_filter[0])
        self.conv1_2 = VGGBlock(self.nb_filter[1]*2+self.nb_filter[2], self.nb_filter[1], self.nb_filter[1])
        self.conv2_2 = VGGBlock(self.nb_filter[2]*2+self.nb_filter[3], self.nb_filter[2], self.nb_filter[2])
        # 第三层解码器的KAN块
        self.dblock2 = KANBlock(dim=self.nb_filter[2], norm_layer=nn.LayerNorm)
        # 更深层的嵌套连接
        self.conv0_3 = VGGBlock(self.nb_filter[0]*3+self.nb_filter[1], self.nb_filter[0], self.nb_filter[0])
        self.conv1_3 = VGGBlock(self.nb_filter[1]*3+self.nb_filter[2], self.nb_filter[1], self.nb_filter[1])
        self.conv0_4 = VGGBlock(self.nb_filter[0]*4+self.nb_filter[1], self.nb_filter[0], self.nb_filter[0])

        # 为x0_1, x0_2, x0_3, x0_4特征图创建CBAM模块
        self.cbam_x0_1 = CBAMBlock(channel=self.nb_filter[0], reduction=16, kernel_size=3)
        self.cbam_x0_2 = CBAMBlock(channel=self.nb_filter[0], reduction=16, kernel_size=3)
        self.cbam_x0_3 = CBAMBlock(channel=self.nb_filter[0], reduction=16, kernel_size=3)
        self.cbam_x0_4 = CBAMBlock(channel=self.nb_filter[0], reduction=16, kernel_size=3)

        # 输出层
        if self.deep_supervision:
            # 如果使用深度监督，为每个解码器输出创建一个卷积层
            self.final1 = nn.Conv2d(self.nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(self.nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(self.nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(self.nb_filter[0], num_classes, kernel_size=1)
        else:
            # 否则只为最终输出创建一个卷积层
            self.final = nn.Conv2d(self.nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        """
        前向传播函数
        实现UNet++的前向传播，并在深层特征中应用KAN模块
        """
        # 编码器路径
        # 第一层特征
        x0_0 = self.conv0_0(input)
        # 第二层特征
        x1_0 = self.conv1_0(self.pool(x0_0))
        # 第一层解码器特征
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        # 应用CBAM到x0_1
        x0_1 = self.cbam_x0_1(x0_1)

        # 第三层特征（经过KAN处理）
        x2_0_raw = self.conv2_0(self.pool(x1_0))
        # 新增：将x2_0转换为序列表示，应用KAN块
        out, H, W = self.patch_embed2(x2_0_raw)
        out = self.block2(out, H, W)
        # 将序列表示转换回空间表示
        x2_0 = out.reshape(-1, H, W, self.nb_filter[2]).permute(0, 3, 1, 2).contiguous()
        
        # 第二层解码器特征
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        # 第一层深度解码器特征
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        # 应用CBAM到x0_2
        x0_2 = self.cbam_x0_2(x0_2)

        # 第四层特征
        x3_0 = self.conv3_0(self.pool(x2_0))
        # 将x3_0转换为序列表示，应用KAN块
        out, H, W = self.patch_embed3(x3_0)
        out = self.block3(out, H, W)
        # 将序列表示转换回空间表示
        x3_0 = out.reshape(-1, H, W, self.nb_filter[3]).permute(0, 3, 1, 2).contiguous()

        # 第五层特征
        # 将x3_0转换为序列表示，应用KAN块
        out, H, W = self.patch_embed4(x3_0)
        out = self.block4(out, H, W)
        # 将序列表示转换回空间表示
        x4_0 = out.reshape(-1, H, W, self.nb_filter[4]).permute(0, 3, 1, 2).contiguous()

        # 第四层解码器特征
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        # 将x3_1转换为序列表示，应用KAN块
        _, _, H, W = x3_1.shape
        out = x3_1.flatten(2).transpose(1, 2)
        out = self.dblock3(out, H, W)
        # 将序列表示转换回空间表示
        x3_1 = out.reshape(-1, H, W, self.nb_filter[3]).permute(0, 3, 1, 2).contiguous()

        # 修复尺寸不匹配问题：对x3_1进行两次上采样
        # 从14x14上采样到28x28，再到56x56
        x3_1_up = self.up(self.up(x3_1))
        # 第三层解码器特征
        x2_1 = self.conv2_1(torch.cat([x2_0, x3_1_up], 1))
        
        # 第二层深度解码器特征
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        # 第一层更深解码器特征
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        # 应用CBAM到x0_3
        x0_3 = self.cbam_x0_3(x0_3)

        # 第三层深度解码器特征
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, x3_1_up], 1))  # 使用相同的上采样结果
        # 将x2_2转换为序列表示，应用KAN块
        _, _, H, W = x2_2.shape
        out = x2_2.flatten(2).transpose(1, 2)
        out = self.dblock2(out, H, W)
        # 将序列表示转换回空间表示
        x2_2 = out.reshape(-1, H, W, self.nb_filter[2]).permute(0, 3, 1, 2).contiguous()

        # 第二层更深解码器特征
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        # 最终解码器特征
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        # 应用CBAM到x0_4
        x0_4 = self.cbam_x0_4(x0_4)

        # 输出处理
        if self.deep_supervision:
            # 如果使用深度监督，返回所有解码器的输出
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            # 否则只返回最终输出
            output = self.final(x0_4)
            return output 
```


kan.py
```
import torch
import torch.nn.functional as F
import math


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
```


dataset.py
```
import os

import cv2
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        mask = []
        for i in range(self.num_classes):
            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                        img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        #数组沿深度方向进行拼接。
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)#这个包比较方便，能把mask也一并做掉
            img = augmented['image']#参考https://github.com/albumentations-team/albumentations
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        
        return img, mask, {'img_id': img_id}


losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss

```

metrics.py
```
import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output = output > 0.5
    target = target > 0.5
    intersection = (output & target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


def precision_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output = output > 0.5
    target = target > 0.5
    
    true_positives = (output & target).sum()
    predicted_positives = output.sum()

    return (true_positives + smooth) / (predicted_positives + smooth)


def recall_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output = output > 0.5
    target = target > 0.5
    
    true_positives = (output & target).sum()
    actual_positives = target.sum()

    return (true_positives + smooth) / (actual_positives + smooth)
```


utils.py
```
import argparse


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
```


CBAM.py
```
import numpy as np
import torch
from torch import nn
from torch.nn import init



class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output



class CBAMBlock(nn.Module):

    def __init__(self, channel=512,reduction=16,kernel_size=49):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        return out+residual


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    kernel_size=input.shape[2]
    cbam = CBAMBlock(channel=512,reduction=16,kernel_size=kernel_size)
    output=cbam(input)
    print(output.shape)
```

    