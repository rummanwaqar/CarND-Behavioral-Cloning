import warnings
warnings.filterwarnings("ignore")
import time

from dataset_utils import get_samples, generator, get_dataset_names

from keras.models import Sequential
from keras.layers import Lambda, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

NAME = 'cnn-5x3-lr-1e-4{}'.format(int(time.time()))

def get_model():
    model = Sequential()
    # normalize from -1 to 1
    model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    # convolutional layers
    model.add(Conv2D(24, kernel_size=5, strides=(2, 2), padding='valid', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(36, kernel_size=5, strides=(2, 2), padding='valid', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(48, kernel_size=5, strides=(2, 2), padding='valid', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, kernel_size=3, strides=(1, 1), padding='valid', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, kernel_size=3, strides=(1, 1), padding='valid', activation='relu'))
    # fc layers
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=Adam(lr=1e-4))
    print(model.summary())
    return model


def train():
    # setup tensorboard
    tensorboard = TensorBoard(log_dir='.logs/{}'.format(NAME))
    # model checkpoints
    checkpoint = ModelCheckpoint('model-{epoch:02d}.h5', monitor='val_loss', 
                                 verbose=0, save_best_only=True, mode='auto')
    # read data
    datasets = get_dataset_names()
    BATCH_SIZE = 128
    EPOCHS = 8
    training_samples, validation_samples = get_samples(datasets=datasets,
                                                       split=0.2,
                                                       base_url='data',
                                                       all=True)
    training_generator = generator(training_samples, batch_size=BATCH_SIZE)
    validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

    model = get_model()
    model.fit_generator(training_generator, steps_per_epoch=len(training_samples)/BATCH_SIZE,
                        validation_data=validation_generator, validation_steps=len(validation_samples)/BATCH_SIZE,
                        nb_epoch=EPOCHS, callbacks=[tensorboard, checkpoint])
    model.save('model.h5')

if __name__ == '__main__':
    train()
