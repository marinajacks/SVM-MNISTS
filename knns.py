# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 15:31:27 2018

@author: hello
"""

import pylab as pl
import math
from operator import itemgetter

#Print an image in text (pretty neat).
def printImg(img):
    for i in range(0,28):
        col = ""
        for j in range(0,28):
            if img[i*28 + j] == 0:
                col = col + " "
            else:
                col = col + "0 "
        print(col)

#"Distance" between 2 images.
def img_distance(img1, img2):
    result = 0
    for i in range(0,len(img1)):
        result += math.fabs(img1[i]-img2[i])
    return result

#Read the MNIST data.
#Download from: http://yann.lecun.com/exdb/mnist/.
#The format of the files is well documented in the website.
def readMnistData(filename, labels_filename, limit):
    imgs = open(filename, mode='rb')
    labels = open(labels_filename, mode='rb')

    imgs.read(4)
    labels.read(4)
    labels.read(4)

    img_num = int.from_bytes(imgs.read(4), byteorder="big")
    cols = int.from_bytes(imgs.read(4), byteorder="big")
    rows = int.from_bytes(imgs.read(4), byteorder="big")

    lists = [[],[]]

    if img_num > limit and limit != -1:
        img_num = limit

    #For each image convert it into a pixel_list and add it with the image's label to our "lists" list.
    for i in range(0,img_num):
        pixel_list = []
        for j in range(0, rows):
            for k in range(0, cols):
                pixel_list.append(int.from_bytes(imgs.read(1), byteorder="big"))

        label = int.from_bytes(labels.read(1), byteorder="big")
        lists[0].append(pixel_list)
        lists[1].append(label);

    return lists;

#Function which predicts the label of an image.
#Returns the predicted label of the test_img.
def predict(test_img,training_set_list,knn):
    distance_list = []

    #For each image and label in the training set, calculate the distance from the test_img
    #and add the distance and the label in a new list.
    for img, label in zip(training_set_list[0], training_set_list[1]):
        distance_list.append([img_distance(test_img, img), label]);

    #Sort the distance_list according to the distancenothing to seee here
    distance_list.sort(key=itemgetter(0))

    #A vector representing the number of neighbours for each digit.
    labels = [0,0,0,0,0,0,0,0,0,0]

    #Add 1 each time we find a naerby neighbour whose digit is i
    for i in range(0,knn):
        labels[distance_list[i][1]] += 1

    #The result is the index of the max element.
    result = labels.index(max(labels))

    return result

#Number of images used for training (-1 for as many as in file)
training_limit = 6000
#Number of images used for testing (-1 for as many as in file)
test_limit = 100

#Read sets to lists
#These lists contain 2 other lists.
#The 1st one is the list of the images. Each image is a list of pixels.
#The 2nd one is the list of the labels.

#Read training set to list
training_set_list = readMnistData("D:\\project\\SVM-MNISTS\\train_data\\train-images.idx3-ubyte","D:\\project\\SVM-MNISTS\\train_data\\train-labels.idx1-ubyte",training_limit)
print("训练数据读取完成.")
#Reat test set to list
test_set_list = readMnistData("D:\\project\\SVM-MNISTS\\test_data\\t10k-images.idx3-ubyte","D:\\project\\SVM-MNISTS\\test_data\\t10k-labels.idx1-ubyte",test_limit)
print("测试数据读取完成.")

#The number of nearest neighbours used for the prediction
knn = 5

right = 0
wrong = 0

#For each img and label in the test_set_list, get the result of the prediction
#from the predict function.
for img,label in zip(test_set_list[0], test_set_list[1]):
    result = predict(img, training_set_list, knn)
    if result == label:
        right += 1
    else:
        wrong += 1

print("Right: " + str(right) + " Wrong: " + str(wrong))
