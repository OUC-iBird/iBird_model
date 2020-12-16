import torchvision.models as models
from overrides import overrides
import torch.nn as nn
import torch
from Utils.loader import CustomDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Utils.trainer import Trainer, run_epochs_for_loop
import os
from torchvision import transforms


class BilinearModel(nn.Module):
    """Load model with pretrained weights and initialise new layers."""

    def __init__(self, num_classes: int = 200, pretrained=True) -> None:
        """Load pretrained model, set new layers with specified number of layers."""
        super(BilinearModel, self).__init__()
        model: nn.Module = models.vgg16(pretrained)
        self.features: nn.Module = nn.Sequential(*list(model.features)[:-1])
        self.classifier: nn.Module = nn.Linear(512 ** 2, num_classes)
        self.dropout: nn.Module = nn.Dropout(0.5)
        self.softmax: nn.Module = nn.LogSoftmax(dim=1)
        nn.init.kaiming_normal_(self.classifier.weight.data)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias.data, val=0)

    @overrides
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Extract input features, perform bilinear transform, project to # of classes and return."""
        outputs: torch.Tensor = self.features(inputs)               # extract features from pretrained base
        outputs = outputs.view(-1, 512, 28 ** 2)                    # reshape to batchsize * 512 * 28 ** 2
        outputs = self.dropout(outputs)
        outputs = torch.bmm(outputs, outputs.permute(0, 2, 1))      # bilinear product
        outputs = torch.div(outputs, 28 ** 2)                       # divide by 196 to normalize
        outputs = outputs.view(-1, 512 ** 2)                        # reshape to batchsize * 512 * 512
        outputs = torch.sign(outputs) * torch.sqrt(outputs + 1e-5)  # signed square root normalization
        outputs = nn.functional.normalize(outputs, p=2, dim=1)      # l2 normalization
        outputs = self.dropout(outputs)
        outputs = self.classifier(outputs)
        # outputs = self.softmax(outputs)
        return outputs

    def load(self, path, complexity=False):
        # 加载模型
        if not complexity:
            # 简易保存模式
            dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.load_state_dict(torch.load(path, map_location=dev))
            print("Load OK")
        else:
            # 复杂保存模式
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model'])
            self.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint(['epoch'])
            acc = checkpoint(["accuracy"])
            print("Load Ok, accuracy={.4f}".format(acc))


if __name__ == '__main__':
    data_dir = "../AI研习社_鸟类识别比赛数据集"
    selection = ["train_set", "val_set"]
    train_labels = os.path.join(data_dir, "train_pname_to_index.csv")
    valid_labels = os.path.join(data_dir, "val_pname_to_index.csv")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using gpu: %s" % torch.cuda.is_available())

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    formats = {
        'train_set': [
            transforms.Resize(448),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.RandomCrop(448),
            transforms.ToTensor(),
            normalize,
        ],
        'val_set': [
            transforms.Resize(448),
            transforms.CenterCrop(448),
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
    }
    model = BilinearModel(num_classes=200)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    criterion.to(device)
    for param in model.features.parameters():
        param.requires_grad = False
    lr = 0.001
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)

    trainer = Trainer(model, criterion, optimizer, device)
    reduceLROnPlateau = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.1,
        patience=10,
        verbose=True,
        threshold=1e-4,
    )
    # 训练 1 轮仅供测试
    run_epochs_for_loop(trainer, 1,
                        data_loader["train"],
                        data_loader["valid"],
                        reduceLROnPlateau)
    for param in model.features.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    reduceLROnPlateau = ReduceLROnPlateau(optimizer, mode='max')
    run_epochs_for_loop(trainer, 1,
                        data_loader["train"],
                        data_loader["valid"],
                        reduceLROnPlateau)