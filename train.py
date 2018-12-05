from dataset_utils import get_samples, generator

def train():
    # read data
    datasets = ['data_1', 'data_2', 'data_3']
    training_samples, validation_samples = get_samples(datasets=datasets,
                                                       split=0.2,
                                                       base_url='data')
    training_generator = generator(training_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)


if __name__ == '__main__':
    train()
