from keras.layers import Input,Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from MachineLearning.Autoencode.tool import get_dataSets, show

class multilayer(object):

    def create(self):
        x_train, x_test = get_dataSets()

        input_img = Input(shape=(784,))
        encoded = Dense(128, activation='relu')(input_img)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dense(32, activation='relu')(encoded)

        decoded = Dense(64, activation='relu')(encoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(784, activation='sigmoid')(decoded)

        autoencoder = Model(input=input_img, output=decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        autoencoder.fit(x_train, x_train,
                        epochs=1,
                        batch_size=128,
                        shuffle=True,
                        validation_data=(x_test, x_test))
        return autoencoder
    pass

if __name__ == '__main__':
    x_train, x_test = get_dataSets()
    multilayers = multilayer()
    autoencode = multilayers.create()
    decoded_imgs = autoencode.predict(x_test)
    show(x_test, decoded_imgs)