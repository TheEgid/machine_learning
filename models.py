import os
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adadelta, Adam
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization


def build_logistics_regression_model(class_numbers=10):
    model = Sequential()
    model.name = 'logistics_regression_model'
    model.add(Dense(class_numbers, input_shape=(784,), activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model


def build_5_layers_adadelta_optim_model(class_numbers=10):
    model = Sequential()
    model.name = '5_layers_adadelta_optim_model'
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(class_numbers, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adadelta(),
                  metrics=['accuracy'])
    return model


def build_convolutional_model(class_numbers=10):
    model = Sequential()
    model.name = 'convolutional_model'
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu',
                     input_shape=[28, 28, 1]))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(class_numbers, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    return model


def build_new_convolutional_model(class_numbers=10):
    model = Sequential()
    model.name = 'new_convolutional_model'
    model.add(Conv2D(filters=64, kernel_size=3, strides=1,
                     padding='same', activation='relu',
                     input_shape=[28, 28, 1]))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    model.add(Conv2D(filters=128, kernel_size=3, strides=1,
            padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    model.add(Conv2D(filters=256, kernel_size=3, strides=1,
            padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(class_numbers, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    return model