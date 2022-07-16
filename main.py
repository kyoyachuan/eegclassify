from torch import dropout
from utils.dataloader import read_bci_data
from eegclassify.dataset import BCIDataContainer, gen_loader
from eegclassify.trainer import Trainer


def main():
    dataset = BCIDataContainer(*read_bci_data())
    train_loader, test_loader = gen_loader(dataset)

    trainer = Trainer('EEGNet', 'leaky_relu', 'cross_entropy', 'adam', 0.01, 0.00001,
                      spatial_filter_depth=2, temporal_filter_size_1=32, temporal_filter_size_2=64)
    print(trainer.model.summary(input_size=dataset.train_x.shape[1:]))

    trainer.train(train_loader, test_loader, epochs=300)


if __name__ == '__main__':
    main()
