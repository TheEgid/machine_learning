from sklearn.model_selection import train_test_split
import pandas as pd
from time import time
import timeit
from datetime import datetime, timedelta
from keras import utils
from keras.models import Sequential
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import Adadelta, Adam
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dense, Dropout


def differ_time(tm):
    sec = timedelta(seconds=int(tm))
    d = datetime(1, 1, 1) + sec
    differ_time_val = "elapsed time: DAYS:{} HOURS:{} MIN:{} SEC:{}".format(
        d.day-1,
        d.hour,
        d.minute,
        d.second)
    return differ_time_val


def separate_data(path_to_data):
    df = pd.read_csv(path_to_data)
    y = df['label'].values
    X = df[df.columns[1:]].values
    return X, y


def get_fashionmnist_data():
    X_train, y_train = separate_data(r'G:\fashionmnist\fashion-mnist_train.csv')
    X_train = X_train.astype('float32') / 255
    y_train = utils.to_categorical(y_train)

    X_test, y_test = separate_data(r'G:\fashionmnist\fashion-mnist_test.csv')
    X_test = X_test.astype('float32') / 255
    y_test = utils.to_categorical(y_test)
    return X_test, y_test, X_train, y_train


def build_logistics_regression_model(class_numbers=10):
    model = Sequential()
    model.name = 'logistics_regression_model'
    model.add(Dense(class_numbers, input_shape=(784,), activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model


def build_8_layers_adam_optim_model(class_numbers=10):
    model = Sequential()
    model.name = '8_layers_adam_optim_model'
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dense(416, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(48, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(class_numbers, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
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
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=[28, 28, 1]))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Conv2D(filters=512, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(class_numbers, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    return model


def fit_and_calculate_model_score(model, data):
    test_x, test_y, train_x, train_y = data
    if model.name != 'convolutional_model':
        epochs = 40
    else:
        epochs = 10
        train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)  # 28x28
        test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)  # 28x28
    epochs_loss_limit = epochs
    board = TensorBoard(write_graph=True,
                        histogram_freq=0,
                        log_dir='tenzor_logs/{}'.format(time()))
    earlystopping = EarlyStopping(monitor='val_loss',
                                  patience=epochs_loss_limit)
    x_no, x_validation, y_no, y_validation = train_test_split(train_x, train_y,
                                                              test_size=0.33,
                                                              random_state=42)
    model.fit(train_x, train_y,
              epochs=epochs,
              verbose=1,
              batch_size=64,
              validation_data=(x_validation, y_validation),
              callbacks=[board, earlystopping])
    return model.evaluate(test_x, test_y, verbose=0)[1]


if __name__ == '__main__':
    start_time1 = timeit.default_timer()

    X_ts, Y_ts, X_tr, Y_tr = get_fashionmnist_data()
    mydata = X_ts, Y_ts, X_tr, Y_tr

    mymodel1 = build_convolutional_model()
    mymodel2 = build_logistics_regression_model()
    mymodel3 = build_8_layers_adam_optim_model()
    mymodel4 = build_5_layers_adadelta_optim_model()
    #model_list = [mymodel1, mymodel2, mymodel3, mymodel4]

    model_list = [mymodel2]

    results = {}
    for mymodel in model_list:
        start_time = timeit.default_timer()
        score = fit_and_calculate_model_score(mymodel, mydata)
        p1 = str(mymodel.name)
        p2 = "validation accuracy: {0:.3f}".format(score)
        p3 = differ_time(timeit.default_timer() - start_time)
        results.update({p1: {'measure': p2, 'time': p3}})

    max_mod = max(results.keys(), key=lambda k: results[k]['measure'])
    dim_name = results[max_mod]['measure']
    print(results)
    print('Best accurancy - our model: {}: {} '.format(max_mod, dim_name))

    print('Total ' + differ_time(timeit.default_timer() - start_time1))


# {'logistics_regression_model': {'measure': 'validation accuracy: 0.853', 'time': 'Прошло времени: DAYS:0 HOURS:0 MIN:2 SEC:53'}}
# Максимальное accurancy - модель: logistics_regression_model: validation accuracy: 0.853

# 0.856
# 0.897
# 0.928

# logistics_regression_model:Validation accuracy: 0.854:Прошло времени: DAYS:0 HOURS:0 MIN:3 SEC:6
# 8_layers_adam_optim_model:Validation accuracy: 0.899:Прошло времени: DAYS:0 HOURS:0 MIN:14 SEC:28
# 5_layers_adadelta_optim_model:Validation accuracy: 0.906:Прошло времени: DAYS:0 HOURS:0 MIN:14 SEC:8

# 40 -
# logistics_regression_model - validation accuracy: 0.852 - Прошло времени: DAYS:0 HOURS:0 MIN:2 SEC:34
# 8_layers_adam_optim_model - validation accuracy: 0.901 - Прошло времени: DAYS:0 HOURS:0 MIN:12 SEC:43
# 5_layers_adadelta_optim_model - validation accuracy: 0.899 - Прошло времени: DAYS:0 HOURS:0 MIN:12 SEC:44


#python -m tensorboard.main --logdir=tenzor_logs/