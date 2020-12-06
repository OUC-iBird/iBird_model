import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from EfficientNet.attention import ChannelAttention, SpatialAttention


class EfficientNetWithAttention(nn.Module):

    def __init__(self, num_classes: int = 200):
        super(EfficientNetWithAttention, self).__init__()
        self.eff_model = EfficientNet.from_pretrained("efficientnet-b7")
        self._avg_pooling = nn.AdaptiveAvgPool2d(output_size=1)
        self._dropout = nn.Dropout(p=0.5, inplace=False)

        self.fc = nn.Linear(in_features=2560, out_features=num_classes, bias=True)
        self.ca_head = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.ca_tail = ChannelAttention(2560)

    def forward(self, x):
        x = self.eff_model.extract_features(x)
        # 这里出现了点问题, 第一层加不上上去
        # 第一层加入 Attention 机制
        # x = self.ca(x) * x
        # x = self.sa(x) * x

        # x = self.layers(x)

        # 最后一层加入 Attention 机制
        x = self.ca_tail(x) * x
        x = self.sa(x) * x

        x = self._avg_pooling(x)
        if self.eff_model._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self._dropout(x)
            x = self.fc(x)
        return x

