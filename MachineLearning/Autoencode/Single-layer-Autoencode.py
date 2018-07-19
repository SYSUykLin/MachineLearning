from keras.layers import Input,Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from MachineLearning.Autoencode.tool import get_dataSets, show

class single_layer(object):

    def create(self):
        x_train, x_test = get_dataSets()
        encoding_dim = 32
        input_img = Input(shape=(784,))
        encoded = Dense(encoding_dim, activation='relu')(input_img)
        decoded = Dense(784, activation='sigmoid')(encoded)
        autoencoder = Model(input=input_img, output=decoded)
        encoder = Model(input=input_img, output=encoded)
        encoded_input = Input(shape=(encoding_dim,))
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        autoencoder.fit(x_train, x_train,
                        epochs=50,
                        batch_size=128,
                        shuffle=True,
                        validation_data=(x_test, x_test))
        return autoencoder, encoder, decoder
    pass

if __name__ == '__main__':
    x_train, x_test = get_dataSets()
    singel = single_layer()
    autoencode, encoder, decoder = singel.create()
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)
    show(x_test, decoded_imgs)