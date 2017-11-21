from keras.layers import Input, Dense, Activation
from keras.models import Model
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import np_utils
from keras.metrics import binary_accuracy
from keras import backend as K

import numpy as np
np.set_printoptions(linewidth=10000, precision = 3, edgeitems= 100, suppress=True)
import matplotlib.pyplot as plt
plt.ion()


def noop():
    pass

def first_way():

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    data = np.reshape(X_train, (-1, 28*28))
    test_data = np.reshape(X_test, (-1, 28*28))
    labels = np_utils.to_categorical(y_train, 10)  #preprocess labels into one-hot encoddings.
    test_labels = np_utils.to_categorical(y_test, 10).astype('float32')  #preprocess labels into one-hot encoddings.

    # This returns a tensor
    inputs = Input(shape=(784,))

    # a layer instance is callable on a tensor (That's the input)
    # and returns a tensor (the output)
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    model.fit(data, labels)  # starts training

    #predicted_labels = model.predict(test_data)
    scores = model.evaluate(test_data, test_labels)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    #print(K.eval(binary_accuracy(test_labels, predicted_labels)))

    return


def second_way():

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    data = np.reshape(X_train, (-1, 28*28))
    test_data = np.reshape(X_test, (-1, 28*28))
    labels = np_utils.to_categorical(y_train, 10)  #preprocess labels into one-hot encoddings.
    test_labels = np_utils.to_categorical(y_test, 10)  #preprocess labels into one-hot encoddings.

    model = Sequential()
    model.add(Dense(64, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(optimizer='sgd', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    model.fit(data, labels)  # starts training

    #predicted_labels = model.predict(test_data)
    #print(K.eval(binary_accuracy(test_labels, predicted_labels)))

    scores = model.evaluate(test_data, test_labels)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


    return



if __name__ == "__main__":

    #Two ways of building a keras model.
    first_way()
    #second_way()

