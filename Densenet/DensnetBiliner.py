from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from overrides import overrides
import torchvision.models as models

from Utils.loader import CustomDataset
from Utils.trainer import Trainer, run_epochs_for_loop

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class BilinearModel(nn.Module):

    def __init__(self, num_classes: int = 200,
                 channel: int = 1024,
                 fmap_size: int = 9) -> None:
        super(BilinearModel, self).__init__()
        self.channel = channel
        self.fmap_size = fmap_size
        model = models.densenet121(pretrained=True)
        self.features: nn.Module = nn.Sequential(*list(model.children())[:-1])
        self.classifier: nn.Module = nn.Linear(self.channel ** 2, num_classes)
        self.dropout: nn.Module = nn.Dropout(0.5)

        nn.init.xavier_normal_(self.classifier.weight.data)
        if self.classifier.bias is not None:
            torch.nn.init.constant_(self.classifier.bias.data, val=0)

    @overrides
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Extract input features, perform bilinear transform, project to # of classes and return."""
        outputs: torch.Tensor = self.features(inputs)  # extract features from pretrained base
        outputs = outputs.view(-1, self.channel, self.fmap_size ** 2)  # reshape to batchsize * 512 * 28 ** 2
        outputs = F.relu(outputs)
        outputs = self.dropout(outputs)
        outputs = torch.bmm(outputs, torch.transpose(outputs, 1, 2)) / (self.fmap_size ** 2)
        outputs = outputs.view(-1, self.channel ** 2)
        outputs = torch.sign(outputs) * torch.sqrt(outputs + 1e-5)  # signed square root normalization
        outputs = F.normalize(outputs)
        outputs = self.dropout(outputs)
        outputs = self.classifier(outputs)
        return outputs


if __name__ == '__main__':
    img_size = 299
    data_dir = "./AI研习社_鸟类识别比赛数据集"
    selection = ["train_set", "val_set"]
    # pretrained_data_dir = "./imagenet_class_index.json"
    train_labels = os.path.join(data_dir, "train_pname_to_index.csv")
    valid_labels = os.path.join(data_dir, "val_pname_to_index.csv")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    formats = {
        'train_set': [
            transforms.ColorJitter(brightness=0.5),  # 亮度
            transforms.ColorJitter(contrast=0.5),  # 对比度
            transforms.ColorJitter(saturation=0.5),  # 饱和度
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.RandomCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ],
        'val_set': [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
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
    data_loader = {
        "train":
            torch.utils.data.DataLoader(
                data_sets["train_set"],
                batch_size=32,
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
        "fine_tune_train":
            torch.utils.data.DataLoader(
                data_sets["train_set"],
                batch_size=8,
                shuffle=True,
                num_workers=0
            ),
    }
    model = BilinearModel(num_classes=200)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    criterion.to(device)
    lr = 0.005
    for param in model.classifier.parameters():
        param.requires_grad = True
    for param in model.features.parameters():
        param.requires_grad = False
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    trainer = Trainer(model, criterion, optimizer, device)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    run_epochs_for_loop(trainer, 42, data_loader["train"], data_loader["valid"], True, exp_lr_scheduler)

    for param in model.features.parameters():
        param.requires_grad = True
    # reduceLROnPlateau = ReduceLROnPlateau(optimizer, mode='max')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    run_epochs_for_loop(trainer, 10, data_loader["fine_tune_train"], data_loader["valid"], True, exp_lr_scheduler)
