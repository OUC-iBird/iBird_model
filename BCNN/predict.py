from BCNN.VGGBiliner import BilinearModel
from Utils.trainer import Trainer
from torchvision import transforms
import torch
import torch.nn as nn
import os
import torchvision
import csv


data_dir = "../AI研习社_鸟类识别比赛数据集"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using gpu: %s" % torch.cuda.is_available())
model = BilinearModel(num_classes=200)
model.load("../results/checkpoint.pt")

criterion = nn.CrossEntropyLoss()
model.to(device)
criterion.to(device)
lr = 0.001
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)

trainer = Trainer(model, criterion, optimizer, device)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trans = transforms.Compose([
    transforms.Resize((448, 448)),
    # transforms.CenterCrop(448),
    transforms.ToTensor(),
    normalize,
])
predict_sets = torchvision.datasets.ImageFolder(os.path.join(data_dir, "test"), transform=trans)
test_loader = torch.utils.data.DataLoader(
    predict_sets,
    batch_size=1,  # 一次一张一张的预测
    shuffle=False,
    num_workers=0
)
predictions = trainer.predict(test_loader)
answer = []
for index, cls in enumerate(predictions):
    path = predict_sets.imgs[index][0]
    path = path.split("/")
    img_name = path[-1]
    answer.append((img_name, int(predictions[index]) + 1))
answer = sorted(answer, key=lambda x: int(x[0].split(".")[0]))

with open('test.csv', 'w', newline="")as f:
    writer = csv.writer(f)
    for one in answer:
        writer.writerow([one[0], one[1]])
