# iBird-Model

这是中国海洋大学 2020 年软件工程课程设计 iBird 项目的模型代码仓库。

用于实现一个 200 种类鸟类的分类，采用了 BCNN 模型。

更新增加了 EfficientNet 模型

## 仓库构成

```
.
├── README.md
├── ibird.ipynb  # 在 colab 上运行的代码
├── imagenet_class_index.json  # vgg16 迁移模型数据
├── main.py  # 模型训练执行入口
├── model.py  # 模型文件
├── predict.py  # 预测 test 测试集模块
├── test_cuda.py  # 测试 cuda 环境
├── requirements.txt  # 项目依赖环境
├── predict_server.py  # 在服务端运行的代码
├── bird_classes.csv  # 编号对于鸟类的中文名
├── classes.txt  # 编号对于的鸟类英文名
└── trainer.py  # 训练类
```

## 使用说明

### AI 研习社测评

1. 读取依赖: `pip install -r requirements.txt`
2. 前往 AI 研习社下载数据 [200种鸟类分类](https://god.yanxishe.com/4?from=god_home_list)
3. 执行 main.py
4. 执行 predict.py 对 AI 研习社数据进行预测

模型训练的数据将会被放在 result 目录的 checkpoint.pt 里，可以使用 model 的方法进行读取

ps: 自己用的所以写的很烂

### 后端程序执行

ps: 要给队友用所以要好好写，这个分类翻译要整死我了

不确定的: 36, 61, 113, 122, 145, 150

1. 将 model.py, predict_server.py, bird_classes.csv 和计算好保存好的模型文件 model.pth 拷贝到服务器上
2. 在后端代码中添加初始化代码
```python
# 初始化单例
from BCNN.predict_server import NeuralNetwork
model_path = "xxx/xxx/xxx"
classes_path = "xxx/xxx/xxx"
NeuralNetwork.get_instance(model_path, classes_path) # 只有第一次会初始化

# 执行预测
net = NeuralNetwork.get_instance()  #  不再进行初始化, 可以不加参数了
result = net.predicted("xx/xx/xx")
# your code here
```

## 使用 trick

### 数据增强

说着听上去挺高级的，实现其来就是对图片的预处理，包含下面的一些方法:
- 平移：一定尺度内平移
- 旋转：一定角度内旋转
- 翻转：水平或者上下翻转
- 裁剪：在原有图像上裁剪一部分
- 颜色变化：rgb颜色空间进行一些变换（亮度对比度等）
- 噪声扰动：给图像加入一些人工生产的噪声

```python
from torchvision import transforms as transforms
# 随机比例缩放
transforms.Resize((100, 200))
# 随机位置裁剪
transforms.RandomCrop(100)
# 中心裁剪
transforms.CenterCrop(100)
# 随机垂直水平翻转
transforms.RandomVerticalFlip(p=1)
transforms.RandomHorizontalFlip(p=1)   # p表示概率
# 随机角度旋转
transforms.RandomRotation(45)

# 色度，亮度，饱和度，对比度
transforms.ColorJitter(brightness=1)  # 亮度
transforms.ColorJitter(contrast=1)  # 对比度
transforms.ColorJitter(saturation=0.5)  # 饱和度
transforms.ColorJitter(hue=0.5)  # 色度
```