import torch.nn as nn
import torch
import os

from efficientnet_pytorch import EfficientNet
from torch.optim import lr_scheduler
from torchvision import transforms

from EfficientNet.attention import ChannelAttention, SpatialAttention
from Utils.loader import CustomDataset
from Utils.trainer import run_epochs_for_loop, Trainer


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


if __name__ == '__main__':
    data_dir = "../AI研习社_鸟类识别比赛数据集"
    selection = ["train_set", "val_set"]
    train_labels = os.path.join(data_dir, "train_pname_to_index.csv")
    valid_labels = os.path.join(data_dir, "val_pname_to_index.csv")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    formats = {
        'train_set': [
            transforms.Resize(456),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.RandomCrop(456),
            transforms.ToTensor(),
            normalize,
        ],
        'val_set': [
            transforms.Resize(456),
            transforms.CenterCrop(456),
            transforms.ToTensor(),
            normalize,
        ],
    }
    data_label_paths = {
        "train_set": train_labels,
        "val_set": valid_labels,
    }
    data_sets = {}
    for one in selection:
        data_sets[one] = CustomDataset(os.path.join(data_dir, one),
                                       data_label_paths[one],
                                       transforms.Compose(formats[one]))
    # windows操作系统不支持 python 的多进程操作。
    # 而神经网络用到多进程的地方在数据集加载上，
    # 所以将 DataLoader 中的参数 num_workers 设置为 0即可
    data_loader = {
        "train":
            torch.utils.data.DataLoader(
                data_sets["train_set"],
                batch_size=2,
                shuffle=True,
                num_workers=0
            ),

        "valid":
            torch.utils.data.DataLoader(
                data_sets["val_set"],
                batch_size=5,
                shuffle=False,
                num_workers=0
        ),
    }
    model = EfficientNetWithAttention()
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    criterion.to(device)

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(params, lr=0.003, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    trainer = Trainer(model, criterion, optimizer, device)
    run_epochs_for_loop(trainer, 1,
                        data_loader["train"],
                        data_loader["valid"],
                        True,
                        exp_lr_scheduler)

