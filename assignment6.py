# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import os
import numpy as np
np.set_printoptions(linewidth=10000, precision = 3, edgeitems= 100, suppress=True)
import matplotlib.pyplot as plt
plt.ion()
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve


"""
So this LSTM is trained over a body of text to predict the next letter in the text.
It is tested on a validation set by predicting each letter one at a time using the
previous (unrolled) history, and comparing with what's the actual next character.
with this we can calculate and error that is expressed as "perplexity".

Perplexity is a measurement of how well a probability distribution
or probability model predicts a sample.  It may be used to compare probability models.
A low perplexity indicates the probability distribution is good at predicting the sample.
"""

def noop():
    pass

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print(statinfo.st_size)
        raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

filename = maybe_download('text8.zip', 31344016)

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        name = f.namelist()[0]
        data = tf.compat.as_str(f.read(name))
    return data

text = read_data(filename)
print('Data size %d' % len(text))

valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])

vocabulary_size = len(string.ascii_lowercase) + 1 # [a-z] + ' '
first_letter = ord(string.ascii_lowercase[0])

def char2id(char):
    if char in string.ascii_lowercase:
        return ord(char) - first_letter + 1
    elif char == ' ':
        return 0
    else:
        print('Unexpected character: %s' % char)
        return 0

def id2char(dictid):
    if dictid > 0:
        return chr(dictid + first_letter - 1)
    else:
        return ' '

print(char2id('a'), char2id('z'), char2id(' '), char2id('i'))
print(id2char(1), id2char(26), id2char(0))

batch_size=64
num_unrollings=10

class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        self._cursor = [ offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b, char2id(self._text[self._cursor[b]])] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches

def characters(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [id2char(c) for c in np.argmax(probabilities, 1)]

def batches2string(batches):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, characters(b))]
    return s

train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)

print(batches2string(train_batches.next()))
print(batches2string(train_batches.next()))
print(batches2string(valid_batches.next()))
print(batches2string(valid_batches.next()))

def logprob(predictions, labels):
    """
    Log-probability of the true labels in a predicted batch.

    This is like an average (mean) cross entropy.
    """
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
    """Sample one element from a distribution assumed to be an array of normalized
    probabilities.
    """
    r = random.uniform(0, 1)
    s = 0
    for i in range(len(distribution)):
        s += distribution[i]
        if s >= r:
            return i
    return len(distribution) - 1

def sample(prediction):
    """Turn a (column) prediction into 1-hot encoded samples."""
    p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
    p[0, sample_distribution(prediction[0])] = 1.0
    return p

def random_distribution():
    """Generate a random column of probabilities."""
    b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
    return b/np.sum(b, 1)[:,None]


num_nodes = 64

#Define computation graph
graph = tf.Graph()
with graph.as_default():


    #Notice:
    #tf.Variables live in the graph.
    #tf.placeholders will need data to be transfered from the CPU to the graph (running on GPU)
    with tf.variable_scope("LSTM_parameters"):
        # Input gate: input, previous output, and bias.
        wix = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1), name='wix')
        wih = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1), name='wih')
        bi = tf.Variable(tf.zeros([1, num_nodes]), name='bi')

        # Forget gate: input, previous output, and bias.
        wfx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1), name='wfx')
        wfh = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1), name='wfh')
        bf = tf.Variable(tf.zeros([1, num_nodes]), name='bf')

        # Memory cell: input, state and bias.
        wcx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1), name='wcx')
        wch = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1), name='wch')
        bc = tf.Variable(tf.zeros([1, num_nodes]), name='bc')

        # Output gate: input, previous output, and bias.
        wox = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1), name='wox')
        woh = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1), name='woh')
        bo = tf.Variable(tf.zeros([1, num_nodes]), name='bo')

    with tf.variable_scope("saved_parameters"):
        # Variables saving state across unrollings.
        saved_ht = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False, name='saved_ht')
        saved_Ct = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False, name='saved_Ct')

    with tf.variable_scope("output_parameters"):
        # Classifier weights and biases.
        w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1), name='w')
        b = tf.Variable(tf.zeros([vocabulary_size]), name='b')

    # Definition of the cell computation.
    def lstm_cell(xt, ht_1, Ct_1, name):
        """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
        Note that in this formulation, we omit the various connections between the
        previous state and the gates.
        i  is  x(t)
        o is h(t-1)

        """
        with tf.variable_scope(name):
            it = tf.sigmoid(tf.matmul(xt, wix) + tf.matmul(ht_1, wih) + bi)
            ft = tf.sigmoid(tf.matmul(xt, wfx) + tf.matmul(ht_1, wfh) + bf)
            Ct_hat = tf.tanh(tf.matmul(xt, wcx) + tf.matmul(ht_1, wch) + bc)
            Ct = ft * Ct_1 + it * Ct_hat
            ot = tf.sigmoid(tf.matmul(xt, wox) + tf.matmul(ht_1, woh) + bo)
            ht = ot * tf.tanh(Ct)
        return ht, Ct

    # Input data.
    train_data = list()
    for _ in range(num_unrollings + 1):
        train_data.append(tf.placeholder(tf.float32, shape=[batch_size,vocabulary_size]))
    train_inputs = train_data[:num_unrollings]
    train_labels = train_data[1:]  # labels are inputs shifted by one time step.

    # Unrolled LSTM loop.
    with tf.variable_scope("Unrolled_LSTM"):
        hts = list()
        ht = saved_ht
        Ct = saved_Ct
        for i, xt in enumerate(train_inputs):
            ht, Ct = lstm_cell(xt, ht, Ct, name='lstm_cell%i'%i)
            hts.append(ht)

    # State saving across unrollings.
    #control_dependencies makes sure that the variable on the argument are evaluated
    #before the contents inside the context manager.
    with tf.control_dependencies([saved_ht.assign(ht),
                                saved_Ct.assign(Ct)]):
        # Classifier.
        with tf.name_scope("logits"):
            logits = tf.nn.xw_plus_b(tf.concat(hts, 0), w, b)
        with tf.name_scope("loss"):
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.concat(train_labels, 0), logits=logits))

    # Optimizer.
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    #We clip the gradients to avoid gradient exploding.
    #These three lines do what optimizer.minimize(loss) does...
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

    # Predictions.
    with tf.name_scope("train_prediction"):
        train_prediction = tf.nn.softmax(logits)

    with tf.name_scope("sampled_test"):
        # Sampling and validation eval: batch 1, no unrolling.
        sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size], name='sample_input')
        saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]), name='saved_sample_output')
        saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]), name='saved_sample_state')

        #groups multiple graph nodes into one,
        #So that we don't have to tell session.run() to evaluate each one of them.
        #the nodes are grouped into an indicator (here reset_sample_state) and whenever we want to
        #run that part of the graph, we only need to run this indicator.
        reset_sample_state = tf.group(saved_sample_output.assign(tf.zeros([1, num_nodes])),
                                        saved_sample_state.assign(tf.zeros([1, num_nodes])))
        sample_output, sample_state = lstm_cell(sample_input, saved_sample_output, saved_sample_state,
                                                name='lstm_cell_sampled_prediction')
        with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                    saved_sample_state.assign(sample_state)]):
            sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))


num_steps = 7001
summary_frequency = 100

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')


    writer = tf.summary.FileWriter(r"C:\tmp\tb\lstm", graph)


    mean_loss = 0
    for step in range(num_steps):
        batches = train_batches.next()
        feed_dict = dict()
        for i in range(num_unrollings + 1):
            feed_dict[train_data[i]] = batches[i]
        _, l, predictions, lr = session.run(
            [optimizer, loss, train_prediction, learning_rate],
            feed_dict=feed_dict)
        mean_loss += l
        if step % summary_frequency == 0:
            if step > 0:
                mean_loss = mean_loss / summary_frequency
            # The mean loss is an estimate of the loss over the last few batches.
            print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
            mean_loss = 0
            labels = np.concatenate(list(batches)[1:])
            print('Minibatch perplexity: %.2f' % float(np.exp(logprob(predictions, labels))))
            if step % (summary_frequency * 10) == 0:
                # Generate some samples.
                print('=' * 80)
                for _ in range(5):
                    feed = sample(random_distribution())
                    sentence = characters(feed)[0]
                    reset_sample_state.run()
                    for _ in range(79):
                        prediction = sample_prediction.eval({sample_input: feed})
                        feed = sample(prediction)
                        sentence += characters(feed)[0]
                    print(sentence)
                print('=' * 80)
            # Measure validation set perplexity.
            reset_sample_state.run()
            valid_logprob = 0
            for _ in range(valid_size):
                b = valid_batches.next()
                predictions = sample_prediction.eval({sample_input: b[0]})
                valid_logprob = valid_logprob + logprob(predictions, b[1])

            #perplexity is a measurement of how well a probability distribution
            #or probability model predicts a sample.  It may be used to compare probability models.
            #A low perplexity indicates the probability distribution is good at predicting the sample.
            print('Validation set perplexity: %.2f' % float(np.exp(valid_logprob / valid_size)))


noop()