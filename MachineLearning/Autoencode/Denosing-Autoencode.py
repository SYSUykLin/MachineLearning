from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from MachineLearning.Autoencode.tool import get_dataSets, show
from keras.datasets import mnist
import numpy as np

def get_nosing(noise_factor):
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

    noise_factor = noise_factor
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)
    return x_train, x_train_noisy, x_test, x_test_noisy
    pass

class denosing(object):

    def create(self):
        x_train, x_train_noisy, x_test, x_test_noisy = get_nosing(0.5)
        input_img = Input(shape=(28, 28, 1))
        x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(input_img)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
        encoded = MaxPooling2D((2, 2), border_mode='same')(x)

        x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        autoencoder.fit(x_train_noisy, x_train, shuffle=True,epochs=1, batch_size=128,
                        validation_data=(x_test_noisy, x_test))
        return autoencoder
        pass
    pass

if __name__ == '__main__':
    x_train, x_train_noisy, x_test, x_test_noisy = get_nosing(0.5)
    denosing = denosing()
    autoencode = denosing.create()
    decoed_imgs = autoencode.predict(x_test_noisy)
    show(x_test_noisy, decoed_imgs)