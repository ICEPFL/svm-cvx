from __future__ import division
from random import shuffle

import os
import numpy as np

def get_accuracy(true, predicted):
    n = len(true)
    hit = 0
    for i in xrange(n):
        if true[i] == predicted[i]:
            hit += 1
    # print "Hit: ", hit
    # print "Total: ", n
    print hit/n
    return hit/n

def abalone_parser(data_path):
    features = []
    labels = []
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            line = line.split(",")[1:] # remove the gender attribute
            line = [float(item) for item in line]
            if line[-1] >= 15:
                labels.append(1)
            else:
                labels.append(-1)
            line.pop()
            features.append(line)
    f.close()
    return features, labels

def sensorless_parser(data_path):
    features = []
    labels = []
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            line = line.split(" ")
            line = [float(item) for item in line]
            if line[-1] >= 3:
                labels.append(1)
            else:
                labels.append(-1)
            line.pop()
            features.append(line)
    f.close()
    return features, labels

if __name__ == "__main__":
    _, _ = sensorless_parser("./datasets/Sensorless.txt")
