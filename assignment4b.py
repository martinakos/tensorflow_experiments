import os

import numpy as np
np.set_printoptions(linewidth=10000, precision = 3, edgeitems= 100, suppress=True)
#import matplotlib.pyplot as plt
#plt.ion()


from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import tensorflow as tf

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def noop():
    pass

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)


image_size = 28
num_labels = 10
num_channels = 1 # grayscale


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
#train_dataset += 0.5
#valid_dataset += 0.5
#test_dataset += 0.5

def calc_accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])


def lrelu(x, alpha=0.01):
    """
    Leaky relu
    """
    return tf.maximum(x, alpha * x)

def conv_layer(input, channels_in, channels_out, name='conv'):
    """
    Creates a convolutional layer.
    For this function to work well we need to declare the variables with
    get_variable() in a reusable variable_scope, otherwise when the model is recreated
    for validation and test data the variables will be recreated and will have
    different weights, not the trained ones, so it'll look like it's not training.
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        initializer = tf.contrib.layers.variance_scaling_initializer()
        w = tf.get_variable("w", initializer=initializer([5, 5, channels_in, channels_out]) )
        b = tf.get_variable("b", initializer=tf.constant(0.0, shape=[channels_out]))
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        #act = tf.nn.relu(conv + b)
        act = lrelu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
    return act


def fc_layer(input, channels_in, channels_out, name='fc', use_relu=True, use_dropout=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        initializer = tf.contrib.layers.variance_scaling_initializer()
        w = tf.get_variable("w", initializer=initializer([channels_in, channels_out]))
        b = tf.get_variable("b", initializer=tf.constant(0.0, shape=[channels_out]))
        act = tf.matmul(input, w) + b
        if use_dropout:
            act = tf.nn.dropout(act, keep_prob)
        if use_relu:
            #act = tf.nn.relu(act)
            act = lrelu(act)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
    return act



batch_size = 32

with tf.device("/device:GPU:0"):

    graph = tf.Graph()
    with graph.as_default():


        # Input data.
        X = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels), name='X')
        #Just to see in tensorboard that what we are using as imputs are really images.
        #if data wasn't arragent as an image here we can reshape it an add it to the summary.
        tf.summary.image('input', X, 3)
        Y = tf.placeholder(tf.float32, shape=(None, num_labels), name='Y')
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
        keep_prob = tf.placeholder(tf.float32)
        global_step = tf.Variable(0)

        def model(X):
            net = conv_layer(X, 1, 8, 'conv1')
            net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

            net = conv_layer(net, 8, 16, 'conv2')
            net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

            net = conv_layer(net, 16, 32, 'conv3')
            net = tf.reshape(net, [-1, 7*7*32])

            net = fc_layer(net, 7*7*32, 1024, 'fc1')
            logits = fc_layer(net, 1024, 10, 'fc2', False, False)

            return logits


        logits = model(X)

        with tf.variable_scope("fc1", reuse=tf.AUTO_REUSE):
            w3 = tf.get_variable("w")
        with tf.variable_scope("fc2", reuse=tf.AUTO_REUSE):
            w4 = tf.get_variable("w")
        alpha = 0.001
        with tf.name_scope("loss"):
            loss = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
                + alpha * tf.nn.l2_loss(w3)
                + alpha * tf.nn.l2_loss(w4))
            tf.summary.scalar("loss", loss)

        # Optimizer.
        learning_rate = tf.train.exponential_decay(0.01, global_step, 50, 0.98, staircase=True)
        with tf.name_scope("train"):
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            #train_op = optimizer.minimize(loss, global_step=global_step)
            train_op = tf.train.AdamOptimizer().minimize(loss)


        # Predictions for the training, validation, and test data.
        with tf.name_scope("train_prediction"):
            train_prediction = tf.nn.softmax(logits)

            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", accuracy)


        with tf.name_scope("valid_prediction"):
            valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
        with tf.name_scope("test_prediction"):
            test_prediction = tf.nn.softmax(model(tf_test_dataset))




    num_steps = 2001
    with tf.Session(graph=graph,
                    config=tf.ConfigProto(intra_op_parallelism_threads=2,
                                          log_device_placement=True)
                    ) as session:
        tf.global_variables_initializer().run()
        print('Initialized')


        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(r"C:\tmp\tb\tb1", graph)


        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            #feed_dict is a dictionary that uses symbolic Tensors as keys
            #The Tensors have the __hash__() function returning the object Id
            #This allows us to access a key using the object that is stored in that key???!!! funny
            feed_dict = {X : batch_data, Y : batch_labels, keep_prob : 0.5}
            _, l, predictions = session.run([train_op, loss, train_prediction], feed_dict=feed_dict)
            if (step % 50 == 0):
                print("iteration:", step, " learning rate:", learning_rate.eval(), " loss:", l,
                    "accuracy: %.1f%%" % calc_accuracy(predictions, batch_labels),
                    " validation accuracy: %.1f%%" % calc_accuracy(
                    valid_prediction.eval(feed_dict={keep_prob : 1.0}), valid_labels),
                    " test accuracy: %.1f%%" % calc_accuracy(
                    test_prediction.eval(feed_dict={keep_prob : 1.0}), test_labels))
            if step % 50 == 0:
                [train_accuracy, s] = session.run([accuracy, merged_summary], feed_dict=feed_dict)
                writer.add_summary(s, step)



noop()