"""
models.vision — 图像模型：CNN 分类 / ViT 分类

- SimpleResNet : 原平台轻量残差网络（自 model.py 迁入，结构不变）
- ViTModel     : Vision Transformer 分类模型，支持积木式注意力选择
"""
import torch
import torch.nn as nn

from architectures.blocks import TransformerEncoderBlock
from architectures.attention import resolve_attention_plan


class ConvBlock(nn.Module):
    """卷积 + BN + ReLU（自 model.py 迁入）"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.conv(x)
        identity = x if self.downsample is None else self.downsample(x)
        return nn.functional.relu(out + identity)


class SimpleResNet(nn.Module):
    """轻量级ResNet，用于图像分类任务（自 model.py 迁入，结构不变）"""

    def __init__(self, image_size=224, num_classes=2, in_channels=3,
                 base_channels=64, dropout=0.1):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(base_channels, base_channels, 2)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, num_classes)
        )
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [ConvBlock(in_channels, out_channels, stride=stride)] + \
                 [ConvBlock(out_channels, out_channels) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        return self.fc(x)


class ViTModel(nn.Module):
    """
    Vision Transformer 图像分类：
    图片 -> patch 序列（卷积切分）-> [CLS] + 位置编码 -> N 个双向注意力块 -> CLS 向量分类。
    attention_type 可选 full/flash/linear（默认 flash），像搭积木一样在 Web 端选择。
    """

    def __init__(self, image_size=224, patch_size=16, in_channels=3, num_classes=10,
                 d_model=256, n_layers=6, n_heads=8, d_ff=512, dropout=0.1,
                 attention_type='flash', attention_plan=None):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size({image_size}) 必须能被 patch_size({patch_size}) 整除")
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.num_classes = num_classes
        self.attention_type = (attention_type or 'flash').lower()
        # 逐层混合注意力计划（缺失时回退统一类型）
        self.attention_plan = resolve_attention_plan(
            attention_plan, n_layers, default=self.attention_type)
        if len(set(self.attention_plan)) > 1:
            print(f"[ViT] 混合注意力计划({n_layers}层): {self.attention_plan}")
            self.attention_summary = '混合[' + '→'.join(self.attention_plan) + ']'
        else:
            self.attention_summary = self.attention_plan[0]
        self.d_model = d_model

        self.patch_embed = nn.Conv2d(in_channels, d_model,
                                     kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                                    dropout=dropout,
                                    attn_name=self.attention_plan[i])
            for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        b = x.size(0)
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)   # (B, N, D)
        tokens = torch.cat([self.cls_token.expand(b, -1, -1), patches], dim=1)
        h = self.drop(tokens + self.pos_emb)
        for blk in self.blocks:
            h = blk(h)
        cls_h = self.norm(h[:, 0])
        return self.head(cls_h)
