import os

import numpy as np
np.set_printoptions(linewidth=10000, precision = 3, edgeitems= 100, suppress=True)
import matplotlib.pyplot as plt
plt.ion()


from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import tensorflow as tf


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
    dataset = dataset.reshape(
      (-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])


def get_global_variable(name):
    var = [v for v in tf.global_variables() if v.name == name][0]
    return var


batch_size = 16
patch_size = 5
patch_size2 = 5
depth = 20
depth2 = 40
num_hidden = 1000
num_hidden2 = 1000

graph = tf.Graph()

with graph.as_default():

    # Input data.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    keep_prob = tf.placeholder(tf.float32)
    global_step = tf.Variable(0)

    with tf.variable_scope("conv1"):
        w1 = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1), name='w')
        b1 = tf.Variable(tf.zeros([depth]), name='b')
    with tf.variable_scope("conv2"):
        w2 = tf.Variable(tf.truncated_normal([patch_size2, patch_size2, depth, depth2], stddev=0.1), name='w')
        b2 = tf.Variable(tf.constant(1.0, shape=[depth2]), name='b')
    with tf.variable_scope("fc1"):
        w3 = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth2, num_hidden], stddev=0.1), name='w')
        b3 = tf.Variable(tf.constant(1.0, shape=[num_hidden]), name='b')

    with tf.variable_scope("fc2"):
        w4 = tf.Variable(tf.truncated_normal([num_hidden, num_hidden2], stddev=0.1), name='w')
        b4 = tf.Variable(tf.constant(1.0, shape=[num_hidden2]), name='b')

    with tf.variable_scope("fc3"):
        w5 = tf.Variable(tf.truncated_normal([num_hidden2, num_labels], stddev=0.1), name='w')
        b5 = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='b')



    def model(data):
        conv = tf.nn.conv2d(data, w1, [1, 1, 1, 1], padding='SAME')
        conv = tf.nn.relu(conv + b1)
        conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv = tf.nn.conv2d(conv, w2, [1, 1, 1, 1], padding='SAME')
        conv = tf.nn.relu(conv + b2)
        conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        shape = conv.get_shape().as_list()
        reshape = tf.reshape(conv, [-1, shape[1] * shape[2] * shape[3]])
        hidden = tf.matmul(reshape, w3) + b3
        hidden = tf.nn.dropout(hidden, keep_prob)
        hidden = tf.nn.relu(hidden)

        hidden = tf.matmul(hidden, w4) + b4
        hidden = tf.nn.dropout(hidden, keep_prob)
        hidden = tf.nn.relu(hidden)

        output = tf.matmul(hidden, w5) + b5
        return output

    # Training computation.
    logits = model(tf_train_dataset)


    #w_fc1 = get_global_variable("fc1/w:0")
    #w_fc2 = get_global_variable("fc2/w:0")

    alpha = 0.001
    loss = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
            + alpha * tf.nn.l2_loss(w3)
            + alpha * tf.nn.l2_loss(w4))

    # Optimizer.
    learning_rate = tf.train.exponential_decay(0.005, global_step, 50, 0.99, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))


writer = tf.summary.FileWriter(r"C:\tmp\tb\tb1", graph)


num_steps = 2001
with tf.Session(graph=graph,
                config=tf.ConfigProto(intra_op_parallelism_threads=3)
                ) as session:
    tf.global_variables_initializer().run()

    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob : 0.5}
        _, l, predictions = session.run([train_op, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
            print("iteration:", step, " learning rate:", learning_rate.eval(), " loss:", l,
                "accuracy: %.1f%%" % accuracy(predictions, batch_labels),
                " validation accuracy: %.1f%%" % accuracy(
                valid_prediction.eval(feed_dict={keep_prob : 1.0}), valid_labels),
                " test accuracy: %.1f%%" % accuracy(
                test_prediction.eval(feed_dict={keep_prob : 1.0}), test_labels))

    writer = tf.summary.FileWriter(r"X:\Deep Learning\Udacity\scripts\tmp", session.graph)


    noop()