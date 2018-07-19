from keras.layers import Input,Dense
from keras.models import Model
from keras.datasets import mnist
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
from MachineLearning.Autoencode.tool import get_dataSets, show
class sparse_layer(object):

    def create(self):
        x_train, x_test = get_dataSets()
        encoding_dim = 32
        input_img = Input(shape=(784,))
        encoded = Dense(encoding_dim, activation='relu',
                        activity_regularizer=regularizers.l1(10e-5))(input_img)
        decoded = Dense(784, activation='sigmoid')(encoded)
        autoencoder = Model(input=input_img, output=decoded)
        encoder = Model(input=input_img, output=encoded)
        encoded_input = Input(shape=(encoding_dim,))
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        autoencoder.fit(x_train, x_train,
                        epochs=1,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(x_test, x_test))
        return autoencoder, encoder, decoder
    pass
if __name__ == '__main__':
    x_train, x_test = get_dataSets()
    singel = sparse_layer()
    autoencode, encoder, decoder = singel.create()
    decoded_imgs = autoencode.predict(x_test)
    show(x_test, decoded_imgs)