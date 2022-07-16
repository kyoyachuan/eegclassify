from utils.dataloader import read_bci_data
from eegclassify.dataset import BCIDataContainer, gen_loader
from eegclassify.trainer import Trainer


def main():
    dataset = BCIDataContainer(*read_bci_data())
    train_loader, test_loader = gen_loader(dataset)

    trainer = Trainer('EEGNet', 'leaky_relu', 'cross_entropy', 'adam', 0.01)
    # print(trainer.model.summary(input_size=(1, 2, 750)))

    trainer.train(train_loader, test_loader, epochs=300)


if __name__ == '__main__':
    main()
