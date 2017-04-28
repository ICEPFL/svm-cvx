from __future__ import division

import re
import os
import matplotlib.pyplot as plt
import numpy as np

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def visualize(train_data, train_labels):
    posi_train = []
    nega_train = []
    for idx, label in enumerate(train_labels):
        if label > 0.:
            posi_train.append(train_data[idx])
        else:
            nega_train.append(train_data[idx])

    posi_train = np.array(posi_train)
    nega_train = np.array(nega_train)

    plt.plot(posi_train[:, 0], posi_train[:, 1], "ro")
    plt.plot(nega_train[:, 0], nega_train[:, 1], "bo")
    plt.show()

def get_accuracy(true, predicted):
    n = len(true)
    hit = 0
    for i in xrange(n):
        if true[i] == predicted[i]:
            hit += 1
    return hit/n
