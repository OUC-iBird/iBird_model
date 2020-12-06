import torch
import pandas as pd
import torch.nn as nn
from torchvision.datasets.folder import accimage_loader, pil_loader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from EfficientNet.trainer import Trainer, run_epochs_for_loop
from EfficientNet.model import EfficientNetWithAttention
import os
from torchvision import transforms


data_dir = "../AI研习社_鸟类识别比赛数据集"
selection = ["train_set", "val_set"]
train_labels = os.path.join(data_dir, "train_pname_to_index.csv")
valid_labels = os.path.join(data_dir, "val_pname_to_index.csv")


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, data_label_path, data_transform, data_loader=default_loader):
        """

        :param data_path: 要读取的文件的路径
        :param data_label_path: 标签数据的路径
        :param data_transform: 数据变换模式
        :param data_loader: 加载方法
        """
        # 在label文件中注意不要加上标签
        df = pd.read_csv(data_label_path, header=None)
        self.data_loader = data_loader
        self.data_transform = data_transform
        self.data_path = data_path
        # 获取文件夹下的全部图片名
        # self.img_names = os.listdir(os.getcwd())
        # self.labels = [" ".join(img_name.split(".")[1].split("_")[:-1]) for img_name in self.img_names]
        self.img_names = list(df[0])
        self.labels = list(df[1])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, item):
        img_name = self.img_names[item]
        img_path = os.path.join(self.data_path, img_name)
        label = self.labels[item]
        img = self.data_loader(img_path)
        try:
            img = self.data_transform(img)
            return img, label-1
        except:
            raise Exception("cannot transform image: {}".format(img_name))


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    formats = {
        'train_set': [
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            normalize,
        ],
        'val_set': [
            transforms.Resize(224),
            transforms.CenterCrop(224),
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

        "valid": torch.utils.data.DataLoader(
                data_sets["val_set"],
                batch_size=5,
                shuffle=False,
                num_workers=0
            ),
    }
    # model = EfficientNet.from_pretrained("efficientnet-b7")
    # in_features = model.fc.in_features
    # model.fc = torch.nn.Linear(in_features, 200)
    model = EfficientNetWithAttention()
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    criterion.to(device)
    for param in model.fc.parameters():
        param.requires_grad = False
    lr = 0.005
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)

    trainer = Trainer(model, criterion, optimizer, device)
    reduceLROnPlateau = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.1,
        patience=10,
        verbose=True,
        threshold=1e-4,
    )
    run_epochs_for_loop(trainer, 1,
                        data_loader["train"],
                        data_loader["valid"],
                        reduceLROnPlateau)
