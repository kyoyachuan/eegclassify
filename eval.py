import tomli
import torch

from eegclassify.dataset import BCIDataContainer, gen_loader
from eegclassify.model import EEGNet, DeepConvNet
from eegclassify.trainer import get_device
from utils.dataloader import read_bci_data


def eval():
    """
    Evaluate the model.
    """
    with open('eval_config/best.toml', 'rb') as f:
        cfg = tomli.load(f)

    dataset = BCIDataContainer(*read_bci_data())
    _, test_loader = gen_loader(dataset)

    device = get_device()

    if cfg['model']['model_name'] == 'EEGNet':
        model = EEGNet(
            activation=cfg['model']['activation'],
            dropout_prob=cfg['model']['dropout_prob'],
            spatial_filter_depth=cfg['model']['spatial_filter_depth'],
            temporal_filter_size_1=cfg['model']['temporal_filter_size_1'],
            temporal_filter_size_2=cfg['model']['temporal_filter_size_2'],
        )
    elif cfg['model']['model_name'] == 'DeepConvNet':
        model = DeepConvNet(
            activation=cfg['model']['activation'],
            dropout_prob=cfg['model']['dropout_prob'],
            channel_list=cfg['model']['channel_list'],
        )
    model.load_state_dict(torch.load(f"models/{cfg['model']['model_path']}"))

    print(model)
    model.summary(input_size=dataset.train_x.shape[1:])

    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()

            print(f"Best model: {cfg['model']['model_name']}, Accuracy {correct / len(target) * 100:.4f}%")


if __name__ == '__main__':
    eval()
