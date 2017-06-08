from __future__ import division
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from keras.datasets import mnist

import numpy as np
import os
import keras
import helpers

abalone = "./datasets/abalone.txt"
sensorless = "./datasets/Sensorless.txt"

class svm_kernel(object):
    def __init__(self, abalone_path, sensorless_path, train_ratio):
        self.abalone_path = abalone_path
        self.sensorless_path = sensorless_path
        self.train_ratio = train_ratio

    def test(self):
        zero = np.zeros((10, 2))
        ones = np.ones((10, 2))
        X_train = np.concatenate((zero, ones), axis=0)
        y_train = np.concatenate((np.ones((10, 1)), np.zeros((10, 1))), axis=0)

        clf = svm.SVC(kernel="linear")
        clf.fit(X_train, y_train)
        print np.abs(clf.dual_coef_)
        print clf.support_vectors_

    def evaluate(self, train_F, test_F, train_L, test_L, kernel="linear", keep_number=6):
        self.fitting_no_pca(train_F, test_F, train_L, test_L, "linear")
        self.fitting_pca(train_F, test_F, train_L, test_L, "linear", 6)

    def abalone(self):
        features, labels = helpers.abalone_parser(self.abalone_path)
        train_F = features[:int(self.train_ratio * len(labels))]
        test_F  = features[int(self.train_ratio * len(labels)):]
        train_L = labels[:int(self.train_ratio * len(labels))]
        test_L  = labels[int(self.train_ratio * len(labels)):]
        self.evaluate( train_F, test_F, train_L, test_L, kernel="linear", keep_number=6)

    def sensorless(self):
        features, labels = helpers.sensorless_parser(self.sensorless_path)
        train_F = features[:int(self.train_ratio * len(labels))]
        test_F  = features[int(self.train_ratio * len(labels)):]
        train_L = labels[:int(self.train_ratio * len(labels))]
        test_L  = labels[int(self.train_ratio * len(labels)):]
        self.evaluate( train_F, test_F, train_L, test_L, kernel="linear", keep_number=len(train_F[0]))

    def mnist(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        y_train = np.array([1 if item > 4 else 0 for item in y_train])
        y_test  = np.array([1 if item > 4 else 0 for item in y_test])

        X_train = (X_train.astype(np.float32) - 127.5)/127.5 # normalize the input
        X_train = X_train.reshape(60000, 784)[:6000][:]
        y_train = y_train[:6000]

        X_test = X_test.reshape(10000, 784)[:500][:]
        y_test = y_test[:500]

        self.evaluate(X_train, X_test, y_train, y_test, kernel="linear", keep_number=2)

    def fitting_no_pca(self, train_F, test_F, train_L, test_L, kernel="linear"):
        print "Start fitting..."
        clf = svm.SVC(kernel=kernel)
        clf.fit(train_F, train_L)
        alphas = np.abs(clf.dual_coef_)

        # print "-------"
        # print len(clf.support_)
        # print clf.support_vectors_.shape
        # print clf.dual_coef_.shape
        # print len(train_L)
        # print "-------"

        print "Finish fitting, start testing..."
        predicted = clf.predict(test_F)
        print "Accuracy (origin): "
        _ = helpers.get_accuracy(predicted, test_L)

    def fitting_pca(self, train_F, test_F, train_L, test_L, kernel="linear", keep_number=5):
         pca1 = PCA(n_components=keep_number)
         pca2 = PCA(n_components=keep_number)
         train_F_ = pca1.fit_transform(train_F)
         test_F_  = pca2.fit_transform(test_F)
         print "=========== PCA results ============"
         print "Finish dimension reduction, start fitting..."
         clf = svm.SVC(kernel=kernel)
         clf.fit(train_F_, train_L)
         print "Finish fitting, start testing..."
         predicted = clf.predict(test_F_)
         print "Accuracy after dimension reduction: "
         _ = helpers.get_accuracy(predicted, test_L)


if __name__ == "__main__":
    full_obj = svm_kernel(abalone, sensorless, 0.7)
    full_obj.mnist()
