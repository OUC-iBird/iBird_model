from typing import List
from model import BilinearModel
from torchvision import transforms
from torchvision import get_image_backend
from torchvision.datasets.folder import accimage_loader, pil_loader
import pandas as pd
import torch
import threading


# 用于在服务器上运行 BCNN 模型做鸟类的预测代码
# 实现了一个单例模式的预测类


class NeuralNetwork:
    _instance_lock = threading.Lock()

    def __init__(self, model_path: str = None, classes_path: str = None):
        """

        :param model_path: 训练完成的模型文件的路径
        :param classes_path: 类别文件的路径
        """
        self._model = BilinearModel(num_classes=200)
        try:
            self._model.load(model_path, complexity=False)
        except Exception as e:
            print(e)
            print("ModelLoadingFailedError, please check your path")
        # 固定参数
        for param in self._model.features.parameters():
            param.requires_grad = False
        # 图像变换
        self._trans = transforms.Compose([
            transforms.Resize((448, 448)),
            # transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.classes = pd.read_csv(classes_path, header=None)
        self.name = list(self.classes[1])
        self.labels = list(self.classes[0])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @classmethod
    def get_instance(cls, *args, **kwargs):
        if not hasattr(NeuralNetwork, "_instance"):
            with NeuralNetwork._instance_lock:
                if not hasattr(NeuralNetwork, "_instance"):
                    NeuralNetwork._instance = NeuralNetwork(*args, **kwargs)
        return NeuralNetwork._instance

    @staticmethod
    def default_loader(path):
        if get_image_backend() == 'accimage':
            return accimage_loader(path)
        else:
            return pil_loader(path)

    def predicted(self, img_path) -> List[tuple]:
        """

        :param img_path: 要预测的图片的路径
        :return: 预测前五的种类和其对应的可能性
        """
        with torch.no_grad():
            # 锁住 dropout 层等各种数据，使用训练好的值
            self._model.eval()
            try:
                img = NeuralNetwork.default_loader(img_path)
                img = self._trans(img)
            except Exception:
                print("Img in path:{0} transforms Failed!".format(img_path))
                return []
            self._model.to(self.device)
            img.to(self.device)
            outputs = self._model(img)
            print(outputs)
            outputs = list(outputs)
            outputs = outputs[:5]
            print(outputs)


if __name__ == "__main__":
    model_path = "./AI研习社_鸟类识别比赛数据集/model.pth"
    classes_path = "./bird_classes.csv"
    img_path = "./AI研习社_鸟类识别比赛数据集/val_set/001.Black_footed_Albatross_283.jpg"
    net = NeuralNetwork(model_path, classes_path)
    net.predicted(img_path)
