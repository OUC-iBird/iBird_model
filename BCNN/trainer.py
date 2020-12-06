"""Trainer class to abstract rudimentary training loop."""

from typing import Tuple

import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os


class Trainer(object):
    """Trainer class to abstract rudimentary training loop."""

    def __init__(
            self,
            model: Module,
            criterion: Module,
            optimizer: Optimizer,
            device: torch.device) -> None:
        """Set trainer class with model, criterion, optimizer. (Data is passed to train/eval)."""
        super(Trainer, self).__init__()
        self.model: Module = model
        self.criterion: Module = criterion
        self.optimizer: Optimizer = optimizer
        self.device: torch.device = device

    def train(self, loader: DataLoader) -> Tuple[float, float]:
        """Train model using batches from loader and return accuracy and loss."""
        total_loss, total_acc = 0.0, 0.0
        self.model.train()
        try:
            with tqdm(enumerate(loader), total=len(loader), desc='Training') as proc:
                for _, (inputs, targets) in proc:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    _, predicted = torch.max(outputs, 1)
                    total_loss += loss.item()
                    total_acc += (predicted == targets).float().sum().item() / targets.numel()
        except Exception as e:
            # 异常情况关闭
            print("Running Error in training, ", e)
            proc.close()
            return -1, -1
        proc.close()
        return total_loss / len(loader), 100.0 * total_acc / len(loader)

    def test(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model using batches from loader and return accuracy and loss."""
        with torch.no_grad():
            total_loss, total_acc = 0.0, 0.0
            self.model.eval()
            try:
                with tqdm(enumerate(loader), total=len(loader), desc='Testing ') as proc:
                    for _, (inputs, targets) in proc:
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                        _, predicted = torch.max(outputs, 1)
                        total_loss += loss.item()
                        total_acc += (predicted == targets).float().sum().item() / targets.numel()
            except Exception as e:
                proc.close()
                print("Running Error in validating,", e)
                return -1, -1
            proc.close()
        return total_loss / len(loader), 100.0 * total_acc / len(loader)

    def predict(self, loader: DataLoader):
        results = []
        for inputs, targets in loader:
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)
            results.append(predicted)
        return results


def run_epochs_for_loop(
        trainer: Trainer,
        epochs: int,
        train_loader: DataLoader,
        test_loader: DataLoader,
        scheduler: ReduceLROnPlateau = None):
    # Run train + evaluation loop for specified epochs.
    best = 0
    for epoch in range(epochs):
        (train_loss, train_acc) = trainer.train(train_loader)
        (test_loss, test_acc) = trainer.test(test_loader)
        print()
        print("Epoch %d: TrainLoss %f \t TrainAcc %f" % (epoch+1, train_loss, train_acc))
        print("Epoch %d: TestLoss %f \t TestAcc %f" % (epoch+1, test_loss, test_acc))
        # 动态更新学习率
        if scheduler is not None:
            scheduler.step(test_acc)
        # 保存训练的结果
        if test_acc > best:
            save_checkpoint(trainer, epoch, test_acc, "../result")
        break


def save_checkpoint(trainer: Trainer, epoch: int, accuracy: float, path: str):
    # 保存训练结果
    path = os.path.join(path, "checkpoint.pt")
    checkpoint = {
        "model": trainer.model.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "epoch": epoch,
        "accuracy": accuracy,
    }
    torch.save(checkpoint, path)