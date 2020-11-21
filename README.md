# iBird-Model

这是中国海洋大学 2020 年软件工程课程设计 iBird 项目的模型代码仓库。

用于实现一个 200 种类鸟类的分类，采用了 BCNN 模型。

## 仓库构成

```
.
├── READ****ME.md
├── ibird.ipynb  # 在 colab 上运行的代码
├── imagenet_class_index.json  # vgg16 迁移模型数据
├── main.py
├── model.py
├── predict.py
├── test_cuda.py
└── trainer.py
```

## 使用说明

- AI 研习社测评

1. 读取依赖: `pip install -r requirements.txt`
2. 前往 AI 研习社下载数据 [200种鸟类分类](https://god.yanxishe.com/4?from=god_home_list)
3. 执行 main.py
4. 执行 predict.py 对 AI 研习社数据进行预测

模型训练的数据将会被放在 result 目录的 checkpoint.pt 里，可以使用 model 的方法进行读取

ps: 自己用的所以写的很烂

- 后端程序执行

ps: 要给队友用所以要好好写，这个分类翻译要整死我了

不确定的: 36, 61, 113, 122, 145, 150

1. 将 model.py, predict_server.py, bird_classes.csv 和计算好保存好的模型文件 model.pth 拷贝到服务器上
2. 在后端代码中添加初始化代码
```python
# 初始化单例
from predict_server import NeuralNetwork
model_path = "xxx/xxx/xxx"
classes_path = "xxx/xxx/xxx"
NeuralNetwork.get_instance(model_path, classes_path) # 只有第一次会初始化

# 执行预测
net = NeuralNetwork.get_instance()  #  不再进行初始化, 可以不加参数了
result = net.predicted("xx/xx/xx")
# your code here
```