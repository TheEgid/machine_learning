from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import os
import pandas as pd
import random
from keras import utils


def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()
    return img_tensor

def convertMnistData(image):
    img = image.astype('float32')
    img /= 255
    return image.reshape(1,28,28,1)



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


text_labels = ['T-SHIRT', 'TROUSER', 'PULLOVER', 'DRESS', 'COAT', 'SANDAL',
               'SHIRT', 'SNEAKER', 'BAG', 'ANKLE BOOT']


if __name__ == '__main__':
    X_ts, Y_ts, X_tr, Y_tr = get_fashionmnist_data()
    mydata = X_ts, Y_ts, X_tr, Y_tr

    model_path_name = r'tenzor_models/convolutional_model.h5'

    if os.path.isfile(model_path_name):
        random_image_index = random.randint(0, 9999)
        model = load_model(model_path_name)

        image = convertMnistData(X_ts[random_image_index])
        predict = np.argmax(model.predict(image))

        test_values_list = list(Y_ts[random_image_index])
        test_label = test_values_list.index(1.0)

        print('Image: {}, Model: {}, Prediction: {}, Real label: {}'.format(
            random_image_index, model.name,
            text_labels[predict],
            text_labels[test_label]))
    else:
        raise(IOError)






