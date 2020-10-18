'''
Author : F323RED
Date : 2020/10/18
Describe : A class that offer some function to load the MNIST data set.
'''

import gzip
import struct
from array import array

class Loader(object) :
    def __init__(self, train_img_path, train_lab_path, test_img_path, test_lab_path) :
        self.train_img_path = train_img_path
        self.train_lab_path = train_lab_path
        self.test_img_path = test_img_path
        self.test_lab_path = test_lab_path

    # Read images and labels from file
    def ReadImagesLabels(self, image_path, label_path) :
        # Get labels
        labels = []

        # Unzip and open file
        with gzip.open(label_path, "rb") as f :
            # Get magic number(int) and data size(int).
            # They are the first two integers
            # Magic number is for data check.
            # Data size is the number of labels and images
            magic, size = struct.unpack(">II", f.read(8))

            if magic != 2049 :
                raise ValueError("Magic number mismatch, expect 2049, got {0}".format(magic))
            
            labels = array("B", f.read())

        # Get images
        images = []

        with gzip.open(image_path, "rb") as f :
            magic, size, row, colum = struct.unpack(">IIII", f.read(16))

            if magic != 2051 :
                raise ValueError("Magic number mismatch, expect 2051, got {0}".format(magic))

            # size * (28 row * 28 colum) array
            for i in range(size):
                images.append([0] * row * colum)
                images[i][:] = array("B", f.read(row * colum))
            
        return images, labels

    def LoadData(self):
        trainingData = self.LoadTrainingData()
        testData = self.LoadTeasData()
        return trainingData, testData

    def LoadTrainingData(self):
        x_train, y_train = self.ReadImagesLabels(self.train_img_path, self.train_lab_path)

        return (x_train, y_train)

    def LoadTestData(self):
        x_test, y_test = self.ReadImagesLabels(self.test_img_path, self.test_lab_path)

        return (x_test, y_test)