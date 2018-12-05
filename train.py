from dataset_utils import get_samples, generator

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

def get_model():
    model = Sequential()
    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    print(model.summary())
    return model


def train():
    # read data
    datasets = ['data_1', 'data_2', 'data_3']
    training_samples, validation_samples = get_samples(datasets=datasets,
                                                       split=0.2,
                                                       base_url='data')
    training_generator = generator(training_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    model = get_model()
    model.fit_generator(training_generator, samples_per_epoch=len(training_samples),
                        validation_data=validation_generator, nb_val_samples=len(validation_samples),
                        nb_epoch=10)
    model.save('model.h5')

if __name__ == '__main__':
    train()
