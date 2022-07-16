import numpy as np
import torch
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset


class BCIDataContainer:
    def __init__(self, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray):
        """
        BCI data container.

        Args:
            train_x (np.ndarray): training input data x
            train_y (np.ndarray): training data label y
            test_x (np.ndarray): testing input data x
            test_y (np.ndarray): testing data label y
        """
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y


def gen_loader(data_container: BCIDataContainer, batch_size: int = 64) -> DataLoader:
    """
    Generate data loader.

    Args:
        data_container (BCIDataContainer): data container
        batch_size (int, optional): batch size. Defaults to 32.

    Returns:
        DataLoader: data loader
    """
    train_dataset = TensorDataset(
        tensor(data_container.train_x, dtype=torch.float),
        tensor(data_container.train_y, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        tensor(data_container.test_x, dtype=torch.float),
        tensor(data_container.test_y, dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
