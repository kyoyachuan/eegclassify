import torch

from eegclassify.dataset import BCIDataContainer, gen_loader
from eegclassify.trainer import Trainer
from utils.dataloader import read_bci_data
from utils.exp_manager import ExperimentManager, ExperimentCfg


@ExperimentManager('config/exp.toml', plot_within_experiment_group=False)
def eval(cfg: ExperimentCfg) -> dict:
    """
    Evaluation function.

    Args:
        cfg (Experiment): experiment configuration

    Returns:
        dict: dictionary of accuracy results
    """
    dataset = BCIDataContainer(*read_bci_data())
    _, test_loader = gen_loader(dataset, batch_size=cfg.batch_size, use_aug=cfg.use_aug)

    trainer = Trainer(**cfg.trainer)
    trainer.model.load_state_dict(torch.load(f'models/{cfg.name}_{cfg.exp_value}.pt'))
    print(trainer.model)
    trainer.model.summary(input_size=dataset.train_x.shape[1:])

    trainer.validate_epoch(test_loader)

    return {
        'test': trainer.collector.test_acc
    }


if __name__ == '__main__':
    eval()
