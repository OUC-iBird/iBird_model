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
        # 最后一层加入 Attention 机制
        x = self.ca_tail(x) * x
        x = self.sa(x) * x
        x = self._avg_pooling(x)
        if self.eff_model._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self._dropout(x)
            x = self.fc(x)
        return x


"""
x = self._swish(self._bn0(self._conv_stem(x)))
        
        # 第一层加入 Attention 机制?
        x = self.ca_head(x)*x
        x = self.sa(x) * x
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        
        # 最后一层加入 Attention 机制
        x = self.ca_tail(x) * x
        x = self.sa(x) * x
        # Pooling and final linear layer
        x = self._avg_pooling(x)
        
        if self._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self._dropout(x)
            x = self.fc(x)
        return x
"""
