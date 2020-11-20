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

执行 main.py 即可，依赖包括

> tqdm
>
> override
>
> pytorch
>
> torchvision
>
> pandas