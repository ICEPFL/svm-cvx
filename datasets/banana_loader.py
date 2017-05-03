from __future__ import division
from sklearn import svm

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import utils.helpers as helpers

class demo_svm(object):
    '''
    For demo only. Call the svm in the sklearn directly. Change kernels to check proformance.
    We can visualize the results as the data has features of two dimension.
    '''
    def __init__(self, kernel="rbf"):
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
        self.test_ = np.array(test_)
        self._test = np.array(_test)
        self.train_ = np.array(train_)
        self._train = np.array(_train)
        self.kernel = kernel
        print "Finish loading the data. Start fitting..."
        # helpers.visualize(_train, train_)
        self.fitting()

    def fitting(self):
        self.clf = svm.SVC(kernel=self.kernel)
        self.clf.fit(self._train, self.train_)
        print "Finish fitting. Start testing..."
        predicted = self.clf.predict(self._test)
        self.accuracy = helpers.get_accuracy(self.test_, predicted)
        print "Accuracy:", self.accuracy

    def plotting(self):
        h = 0.1
        x_min, x_max = self._train[:, 0].min() - 1, self._train[:, 0].max() + 1
        y_min, y_max = self._train[:, 1].min() - 1, self._train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = self.clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        plt.scatter(self._train[:, 0], self._train[:, 1], c=self.train_, cmap=plt.cm.coolwarm)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.show()

if __name__ == "__main__":
    svm = demo_svm()
    svm.plotting()
