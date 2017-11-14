import os

import numpy as np
np.set_printoptions(linewidth=10000, precision = 3, edgeitems= 100, suppress=True)
import matplotlib.pyplot as plt
plt.ion()


from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import tensorflow as tf


image_size = 28
num_labels = 10
batch_size = 128


def noop():
    pass


def accuracy(predictions, labels):
    acc = (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])
    return acc


def load_and_reformat_data():
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

    def reformat(dataset, labels):
        dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
        #Turn labels in 1-hot encoding.
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
        return dataset, labels

    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    return (train_dataset, train_labels,
            valid_dataset, valid_labels,
            test_dataset, test_labels)



def sgd_1layer():
    (train_dataset, train_labels,
     valid_dataset, valid_labels,
        test_dataset, test_labels) = load_and_reformat_data()


    graph = tf.Graph()
    with graph.as_default():
        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)


        # Variables.
        weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
        biases = tf.Variable(tf.zeros([num_labels]))

        # Training computation.
        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

    num_steps = 3001

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print("Initialized")
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 500 == 0):
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
                print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

    noop()


def sdd_1hidden_layer():
    (train_dataset, train_labels,
     valid_dataset, valid_labels,
     test_dataset, test_labels) = load_and_reformat_data()


    #train_dataset = train_dataset[:batch_size*10,:]
    #train_labels = train_labels[:batch_size*10,:]


    n_input =  image_size * image_size
    n_hidden_1 = 4096
    n_hidden_2 = 512
    n_hidden_3 = 256
    n_classes = num_labels
    alpha = 0.1

    graph = tf.Graph()
    with graph.as_default():

        X = tf.placeholder(tf.float32, shape=(batch_size, n_input))
        Y = tf.placeholder(tf.float32, shape=(batch_size, n_classes))
        keep_prob = tf.placeholder(tf.float32)
        global_step = tf.Variable(0)  # count the number of steps taken.

        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
            'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
            'out': tf.Variable(tf.truncated_normal([n_hidden_3, n_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.zeros([n_hidden_1])),
            'b2': tf.Variable(tf.zeros([n_hidden_2])),
            'b3': tf.Variable(tf.zeros([n_hidden_3])),
            'out': tf.Variable(tf.zeros([n_classes]))
        }

        def multilayer_perceptron(x):
            layer_1 = tf.nn.relu(x @ weights['h1'] + biases['b1'])
            drop_out_1 = tf.nn.dropout(layer_1, keep_prob)
            layer_2 = tf.nn.relu(drop_out_1 @ weights['h2'] + biases['b2'])
            drop_out_2 = tf.nn.dropout(layer_2, keep_prob)
            layer_3 = tf.nn.relu(drop_out_2 @ weights['h3'] + biases['b3'])
            drop_out_3 = tf.nn.dropout(layer_3, keep_prob)
            out_layer = drop_out_3 @ weights['out'] + biases['out']
            return out_layer


        all_weights = tf.concat([tf.reshape(weights['h1'], [-1]),
                                 tf.reshape(weights['h2'], [-1]),
                                 tf.reshape(weights['out'], [-1])], axis=0)

        logits = multilayer_perceptron(X)
        loss = (tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
            #+ alpha * (tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['out']))
            + alpha * tf.nn.l2_loss(all_weights)
        )

        #reg1 = tf.nn.l2_loss(all_weights)
        #reg2 = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['out'])

        learning_rate = tf.train.exponential_decay(0.0005, global_step, 100, 0.96, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)

        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(multilayer_perceptron(tf_valid_dataset))
        test_prediction = tf.nn.softmax(multilayer_perceptron(tf_test_dataset))


    num_steps = 3001
    NUM_THREADS = 2
    with tf.Session(graph=graph,
                    config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)
                    ) as session:
        tf.global_variables_initializer().run()
        print("Initialized")
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {X : batch_data, Y : batch_labels, keep_prob : 0.5}
            _, l, predictions = session.run([train_op, loss, train_prediction], feed_dict=feed_dict)
            if (step % 500 == 0):
                print("learning rate:", learning_rate.eval())
                print("Minibatch loss at step %d: %f" % (step, l))
                #print("Minibatch reg1 at step %d: %f" % (step, reg1.eval(feed_dict=feed_dict)))
                #print("Minibatch reg2 at step %d: %f" % (step, reg2.eval(feed_dict=feed_dict)))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(feed_dict={keep_prob : 1.0}), valid_labels))
                print("Test accuracy: %.1f%%" % accuracy(
                    test_prediction.eval(feed_dict={keep_prob : 1.0}), test_labels))

            #_ = session.run([train_op], feed_dict=feed_dict)
            #if (step % 500 == 0):
                #print("Minibatch loss at step %d: %f" % (step, loss.eval(feed_dict=feed_dict)))
                #print("Minibatch accuracy: %.1f%%" % accuracy(train_prediction.eval(feed_dict=feed_dict), batch_labels))
                #print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
                #print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


    noop()



if __name__ == "__main__":

    #sgd_1layer()
    sdd_1hidden_layer()