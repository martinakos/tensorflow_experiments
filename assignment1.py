import os

import numpy as np
np.set_printoptions(linewidth=10000, precision = 3, edgeitems= 100, suppress=True)
import matplotlib.pyplot as plt
plt.ion()


from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

data_root = '.'


def noop():
    pass

def l2_norm_cols(x):
    '''
    Returns the l2 norm (Euclidean distance) of the columns of x.
    '''
    return np.sqrt(np.einsum('ij,ij->j', x, x))


def check_data(name, dataset, labels):
    print(name)
    unique = np.unique(labels)
    for label in unique:
        mask = (labels == label)
        print("label:", label, "count:", np.count_nonzero(mask),
              " mean:", np.mean(dataset[mask,:,:]), " std:", np.std(dataset[mask,:,:]))
    return


def check_overlapping(dataset1, dataset2, th=1):
    """
    I compute the cosine distance between each image.
    As the images are zero mean already, and I calculate the dot products
    on the unit lenght versions of the image, this is equivalent to the
    Zero-mean Normalized Cross-Correlation

    Note: for faster performance, maybe dataset1 can be the largest of both datasets.
    """

    dataset1p = np.reshape(dataset1, (dataset1.shape[0],-1))
    dataset2p = np.reshape(dataset2, (dataset2.shape[0],-1))
    #Now the samples in these datasets have length 1
    dataset1n = dataset1p / l2_norm_cols(dataset1p.T)[:,None]
    dataset2n = dataset2p / l2_norm_cols(dataset2p.T)[:,None]

    res =[]
    for i, sample in enumerate(dataset2n):
        ds = np.einsum('ji,i->j', dataset1n, sample)
        matches = (ds >= th)
        if matches.any():
            res.append((i, list(np.nonzero(matches)[0])))
        noop()

    print("th:", th, "=> n overlaping images:", len(res))

    return res

def show_samples(dataset1, dataset2, res):

    for r in res:
        plt.figure("dataset2")
        plt.title("im:%i"%r[0])
        plt.imshow(dataset2[r[0],:,:])
        for i in r[1]:
            plt.figure("dataset1")
            plt.title("im:%i"%i)
            plt.imshow(dataset1[i,:,:])
            noop()


def train_classifier(train_dataset, train_labels):

    train_datasetp = np.reshape(train_dataset, (train_dataset.shape[0],-1))

    C = 1.0
    clf = LogisticRegression(C=C, solver='lbfgs', multi_class='multinomial')
    clf.fit(train_datasetp, train_labels)
    return clf


def test_classifier(clf, test_dataset, test_labels):
    from sklearn.metrics import classification_report
    test_datasetp = np.reshape(test_dataset, (test_dataset.shape[0],-1))

    ys = clf.predict(test_datasetp)
    print(classification_report(test_labels, ys))

    return


if __name__ == "__main__":
    pickle_file = os.path.join(data_root, 'notMNIST.pickle')
    f = open(pickle_file, 'rb')
    data = pickle.load(f)

    #check_data("valid", data['valid_dataset'], data['valid_labels'])
    #check_data("test", data['test_dataset'], data['test_labels'])
    #check_data("train", data['train_dataset'], data['train_labels'])


    #res = check_overlapping(data['valid_dataset'], data['test_dataset'], th=0.95)
    #show_samples(data['valid_dataset'], data['test_dataset'], res)
    #check_overlapping(data['train_dataset'], data['test_dataset'])

    N = data['train_dataset'].shape[0]
    clf = train_classifier(data['train_dataset'][:N,:,:],
                     data['train_labels'][:N])

    test_classifier(clf, data['valid_dataset'], data['valid_labels'])
    test_classifier(clf, data['test_dataset'], data['test_labels'])


    print()