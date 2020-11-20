# iBird-Model

这是中国海洋大学 2020 年软件工程课程设计 iBird 项目的模型代码仓库。

用于实现一个 200 种类鸟类的分类，采用了 BCNN 模型。

## 仓库构成

```
.
├── README.md
├── ibird.ipynb  # 在 colab 上运行的代码
├── imagenet_class_index.json  # vgg16 迁移模型数据
├── main.py
├── model.py
├── predict.py
├── test_cuda.py
└── trainer.py
```

## 使用说明

1. 读取依赖: `pip install -r requirements.txt`
2. 前往 AI 研习社下载数据 [200种鸟类分类](https://god.yanxishe.com/4?from=god_home_list)
3. 执行 main.py
4. 执行 predict.py 对 AI 研习社数据进行预测

模型训练的数据将会被放在 result 目录的 checkpoint.pt 里，可以使用 model 的方法进行读取