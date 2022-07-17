from eegclassify.dataset import BCIDataContainer, gen_loader
from eegclassify.trainer import Trainer
from utils.dataloader import read_bci_data
from utils.exp_manager import ExperimentManager, ExperimentCfg


@ExperimentManager('config/exp.toml')
def main(cfg: ExperimentCfg) -> dict:
    """
    Main function.

    Args:
        cfg (Experiment): experiment configuration

    Returns:
        dict: dictionary of accuracy results
    """
    dataset = BCIDataContainer(*read_bci_data())
    train_loader, test_loader = gen_loader(dataset)

    trainer = Trainer(**cfg.trainer)
    print(trainer.model.summary(input_size=dataset.train_x.shape[1:]))

    trainer.train(train_loader, test_loader, epochs=cfg.epochs)

    return {
        'train': trainer.collector.train_acc,
        'test': trainer.collector.test_acc
    }


if __name__ == '__main__':
    main()
