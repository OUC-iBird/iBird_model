import os
import pandas as pd
import torch
from torchvision.datasets.folder import accimage_loader, pil_loader


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