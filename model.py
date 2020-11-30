import torchvision.models as models
from overrides import overrides
import torch.nn as nn
import torch


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

    def save(self, path):
        # 只保存模型的实例变量
        if torch.__version__ == "1.6.0":
            torch.save(self.state_dict(), path, _use_new_zipfile_serialization=False)
        else:
            torch.save(self.state_dict(), path)
        print("Save OK")

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
