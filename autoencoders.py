from keras.layers import Input, Dense
from keras.models import Model

from keras.datasets import mnist

import numpy as np
np.set_printoptions(linewidth=10000, precision = 3, edgeitems= 100, suppress=True)
import matplotlib.pyplot as plt
plt.ion()


#We create 3 models. An autoencoder and encoder and a decoder.

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
# a layer instance (Dense) is callable on a tensor, and returns a tensor
encoded = Dense(encoding_dim, activation='relu')(input_img)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

#autoencoder.summary()
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #
#=================================================================
#input_1 (InputLayer)         (None, 784)               0
#_________________________________________________________________
#dense_1 (Dense)              (None, 32)                25120
#_________________________________________________________________
#dense_2 (Dense)              (None, 784)               25872
#=================================================================
#Total params: 50,992
#Trainable params: 50,992
#Non-trainable params: 0
#_________________________________________________________________



# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

#encoder.summary()
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #
#=================================================================
#input_1 (InputLayer)         (None, 784)               0
#_________________________________________________________________
#dense_1 (Dense)              (None, 32)                25120
#=================================================================
#Total params: 25,120
#Trainable params: 25,120
#Non-trainable params: 0
#_________________________________________________________________


# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

#decoder.summary()
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #
#=================================================================
#input_2 (InputLayer)         (None, 32)                0
#_________________________________________________________________
#dense_2 (Dense)              (None, 784)               25872
#=================================================================
#Total params: 25,872
#Trainable params: 25,872
#Non-trainable params: 0
#_________________________________________________________________


autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


#We don't need the labels as the autoencoder is self-supervised

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


#note: x_train, x_train :)
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


#Testing the Autoencoder
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

