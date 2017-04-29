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
clf = svm.SVC(kernel='rbf')
clf.fit(_train, train_)

print "Finish fitting. Start testing..."
predicted = clf.predict(_test)
accuracy = helpers.get_accuracy(test_, predicted)
print "Accuracy:", accuracy

''' plotting the decision area '''

h = 0.2
# create a mesh to plot in
x_min, x_max = _train[:, 0].min() - 1, _train[:, 0].max() + 1
y_min, y_max = _train[:, 1].min() - 1, _train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot also the training points
plt.scatter(_train[:, 0], _train[:, 1], c=train_, cmap=plt.cm.coolwarm)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()
