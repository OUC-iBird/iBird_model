import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using gpu: %s" % torch.cuda.is_available())