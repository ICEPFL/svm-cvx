from __future__ import division
from sklearn import svm

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import utils.helpers as helpers

_train = []
train_ = []

_test = []
test_ = []

for file_name in sorted(glob.glob("./banana/*.asc"), key=helpers.numericalSort):
    temp = np.loadtxt(file_name, skiprows=6)
    if "train_data" in file_name:
        _train.extend(temp)
    if "train_labels" in file_name:
        train_.extend(temp)
    if "test_data" in file_name:
        _test.extend(temp)
    if "test_labels" in file_name:
        test_.extend(temp)

del temp
test_ = np.array(test_)
_test = np.array(_test)

train_ = np.array(train_)
_train = np.array(_train)
# helpers.visualize(_train, train_)
print "Finish loading the data. Start fitting..."
clf = svm.SVC()
clf.fit(_train, train_)

print "Finish fitting. Start testing..."
predicted = clf.predict(_test)
accuracy = helpers.get_accuracy(test_, predicted)

print "Accuracy:", accuracy
