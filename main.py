import os
import random
from time import time
import timeit
from datetime import datetime, timedelta
from keras import utils
from keras.models import load_model
from keras.callbacks import TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from models import build_5_layers_adadelta_optim_model
from models import build_logistics_regression_model
from models import build_convolutional_model
from models import build_new_convolutional_model


def differ_time(tm):
    sec = timedelta(seconds=int(tm))
    d = datetime(1, 1, 1) + sec
    differ_time_val = "elapsed time: DAYS:{} HOURS:{} MIN:{} SEC:{}".format(
        d.day-1,
        d.hour,
        d.minute,
        d.second)
    return differ_time_val


# def load_image(img_path, show=False):
#     img = image.load_img(img_path, target_size=(150, 150))
#     img_tensor = image.img_to_array(img)
#     img_tensor = np.expand_dims(img_tensor, axis=0)
#     img_tensor /= 255.
#     if show:
#         plt.imshow(img_tensor[0])
#         plt.axis('off')
#         plt.show()
#     return img_tensor


def convertMnistData(image):
    img = image.astype('float32')
    img /= 255
    return image.reshape(1,28,28,1)


def separate_data(path_to_data):
    df = pd.read_csv(path_to_data)
    y = df['label'].values
    X = df[df.columns[1:]].values
    return X, y


def show_image(X_ts, count):
    s_image = X_ts[count]
    s_image = np.array(s_image, dtype='float')
    pixels = s_image.reshape((28, 28))
    plt.imshow(pixels, cmap='cool')
    plt.show()


def get_fashionmnist_data():
    X_train, y_train = separate_data(r'G:\fashionmnist\fashion-mnist_train.csv')
    X_train = X_train.astype('float32') / 255
    y_train = utils.to_categorical(y_train)

    X_test, y_test = separate_data(r'G:\fashionmnist\fashion-mnist_test.csv')
    X_test = X_test.astype('float32') / 255
    y_test = utils.to_categorical(y_test)
    return X_test, y_test, X_train, y_train


def fit_save_model_get_score(model, data):
    test_x, test_y, train_x, train_y = data
    if model.name not in ['new_convolutional_model', 'convolutional_model']:
        epochs = 40
    else:
        epochs = 10
        train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)  # 28x28
        test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)  # 28x28
    epochs_loss_limit = epochs
    board = TensorBoard(write_graph=True,
                        histogram_freq=0,
                        log_dir=r'tenzor_logs/{}'.format(time()))
    earlystopping = EarlyStopping(monitor='val_loss',
                                  patience=epochs_loss_limit)
    x_no, x_validation, y_no, y_validation = train_test_split(train_x, train_y,
                                                              test_size=0.33,
                                                              random_state=42)
    model_path_name = r'tenzor_models/{}.h5'.format(model.name)
    if os.path.isfile(model_path_name):
        print(model.name + 'already exists, opening from file')
        model = load_model(model_path_name)  # load_model_hdf5
    else:
        print(model.name + 'does not exist, fit!')
        model.fit(train_x, train_y,
                  epochs=epochs,
                  verbose=1,
                  batch_size=64,
                  validation_data=(x_validation, y_validation),
                  callbacks=[board, earlystopping])
        if os.path.exists == False:
            os.mkdir(r'tenzor_models/')
        model.save(model_path_name)
    return model.evaluate(test_x, test_y, verbose=0)[1]


def get_validation_accuracy(model_list, X_ts, Y_ts, X_tr, Y_tr):
    results = {}
    data = X_ts, Y_ts, X_tr, Y_tr
    for model in model_list:
        #print(model.summary())
        start_time = timeit.default_timer()
        score = fit_save_model_get_score(model, data)
        var1 = str(model.name)
        var2 = "validation accuracy: {0:.3f}".format(score)
        var3 = differ_time(timeit.default_timer() - start_time)
        results.update({var1: {'measure': var2, 'time': var3}})
    max_mod = max(results.keys(), key=lambda k: results[k]['measure'])
    dim_name = results[max_mod]['measure']
    for k, v in results.items():
        print('Model: {}, {}, {}'.format(k, v['time'], v['measure']))
    print('Best accurancy - model: {}: {} '.format(max_mod, dim_name))


def get_prediction_test(model_path_name, X_ts, Y_ts):
    text_labels = ['T-SHIRT', 'TROUSER', 'PULLOVER', 'DRESS', 'COAT', 'SANDAL',
                   'SHIRT', 'SNEAKER', 'BAG', 'ANKLE BOOT']
    if os.path.isfile(model_path_name):
        random_image_index = random.randint(0, 9999)
        model = load_model(model_path_name)
        image = convertMnistData(X_ts[random_image_index])
        predict = int(np.argmax(model.predict(image)))

        test_values_list = list(Y_ts[random_image_index])
        test_label = test_values_list.index(1.0)
        print('Image №: {}, Model: {}, Prediction: {}, Real label: {}'.format(
            random_image_index, model.name,
            text_labels[predict],text_labels[test_label]))
        show_image(X_ts, random_image_index)
    else:
        raise(IOError)


if __name__ == '__main__':
    start_time1 = timeit.default_timer()
    X_ts, Y_ts, X_tr, Y_tr = get_fashionmnist_data()

    mymodel1 = build_convolutional_model()
    mymodel2 = build_logistics_regression_model()
    mymodel3 = build_5_layers_adadelta_optim_model()
    mymodel4 = build_new_convolutional_model()

    models = [mymodel1, mymodel2, mymodel3, mymodel4]

    get_validation_accuracy(models, X_ts, Y_ts, X_tr, Y_tr)
    print('Total ' + differ_time(timeit.default_timer() - start_time1))

    get_prediction_test(r'tenzor_models/new_convolutional_model.h5', X_ts, Y_ts)
