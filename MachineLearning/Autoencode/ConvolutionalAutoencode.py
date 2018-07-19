from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from MachineLearning.Autoencode.tool import get_dataSets, show
from keras.datasets import mnist
import numpy as np
class convolution(object):

    def create(self):
        x_train, x_test = get_dataSets()
        x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))

        x_test = np.reshape(x_test, (len(x_test),28, 28, 1))
        input_img = Input(shape=(28, 28, 1))
        x = Convolution2D(16, (3, 3), activation='relu', border_mode='same')(input_img)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Convolution2D(8, (3, 3), activation='relu', border_mode='same')(x)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Convolution2D(8, (3, 3), activation='relu', border_mode='same')(x)
        encoded = MaxPooling2D((2, 2), border_mode='same')(x)

        x = Convolution2D(8, (3, 3), activation='relu', border_mode='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(8, (3, 3), activation='relu', border_mode='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(16, 3, 3, activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Convolution2D(1, (3, 3), activation='sigmoid', border_mode='same')(x)
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        autoencoder.fit(x_train, x_train, epochs=20, batch_size=256,
                           shuffle=True,
                            validation_data=(x_test, x_test))
        return autoencoder
    pass

if __name__ == '__main__':
    x_train, x_test = get_dataSets()
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
    con = convolution()
    autoencode = con.create()
    decoed_imgs = autoencode.predict(x_test)
    show(x_test, decoed_imgs)


