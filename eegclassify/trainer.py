import torch
from torch import nn
from tqdm import tqdm

from .model import get_model


def get_device() -> torch.device:
    """
    Get the device.

    Returns:
        torch.device: device
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    return device


def get_loss(loss: str) -> nn.Module:
    """
    Get the loss function.

    Args:
        loss (str): string of loss function

    Raises:
        ValueError: We only allow the following losses for experiment:
            - 'cross_entropy'

    Returns:
        nn.Module: loss function
    """
    if loss == 'cross_entropy':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f'Not included loss {loss}')


def get_optimizer(optimizer: str, model: nn.Module, lr: float, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    """
    Get the optimizer.

    Args:
        optimizer (str): string of optimizer
        model (nn.Module): model
        lr (float): learning rate
        weight_decay (float, optional): weight decay. Defaults to 0.0.

    Raises:
        ValueError: We only allow the following optimizers for experiment:
            - 'adam'
            - 'sgd'

    Returns:
        nn.optim.Optimizer: optimizer
    """
    if optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f'Not included optimizer {optimizer}')


class Trainer:
    def __init__(self, model_name: str, activation: str, loss: str, optimizer: str, lr: float, weight_decay: float = 0.0):
        """
        Initialize the trainer.

        Args:
            model_name (str): string of model name
            activation (str): string of activation function
            loss (str): string of loss function
            optimizer (str): string of optimizer
            lr (float): learning rate
            weight_decay (float, optional): weight decay. Defaults to 0.0.
        """
        self.model_name = model_name
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay

        self.device = get_device()
        self.model = get_model(model_name, self.activation).to(self.device)
        self.loss_fn = get_loss(loss)
        self.optimizer = get_optimizer(optimizer, self.model, lr, weight_decay)

    def train(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, epochs: int = 300):
        """
        Train the model.

        Args:
            train_loader (torch.utils.data.DataLoader): train loader
            val_loader (torch.utils.data.DataLoader): validation loader
            epochs (int, optional): number of epochs. Defaults to 300.
        """
        for epoch in range(epochs):
            self.train_epoch(train_loader, epoch)
            self.validate_epoch(val_loader, epoch)

    def train_epoch(self, train_loader: torch.utils.data.DataLoader, epoch: int):
        """
        Train the model for one epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): train loader
            epoch (int): epoch
        """
        self.model.train()
        with tqdm(train_loader, unit='batch') as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f'Epoch {epoch}')
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                accuracy = self._compute_accuracy(output, target)
                loss.backward()
                self.optimizer.step()
                tepoch.set_postfix(loss=loss.item(), accuracy=accuracy)

    def validate_epoch(self, val_loader: torch.utils.data.DataLoader, epoch: int):
        """
        Validate the model for one epoch.

        Args:
            val_loader (torch.utils.data.DataLoader): validation loader
            epoch (int): epoch
        """
        self.model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.loss_fn(output, target).item()
                val_accuracy += self._compute_accuracy(output, target)
        val_loss /= len(val_loader)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {:.4f} / {} ({:.0f}%)\n'.format(
            val_loss, val_accuracy, len(val_loader),
            val_accuracy / len(val_loader)))

    def _compute_accuracy(self, output: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute the accuracy.

        Args:
            output (torch.Tensor): output
            target (torch.Tensor): target

        Returns:
            float: accuracy
        """
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return 100. * correct / len(target)
