# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 15:13:29 2018

@author: hello
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 21:11:37 2018

@author: hello
"""

import binascii
from sklearn import svm
import numpy as np
import matplotlib.pyplot  as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neural_network import MLPClassifier

#iris数据集的数据加载
def loadiris(p):
    f=open(p,'r')
    lines=f.readlines()
    datamat=[]
    for line in lines:
        a=[]
        for i in range(len(line.split(','))-1):
            a.append(float(line.split(',')[i]))
        datamat.append(a)
    return np.array(datamat)
'''获取数据集的标签信息'''       
def loadflags(p):
    f=open(p,'r')
    lines=f.readlines()
    flags=[]
    for line in lines:
        flags.append(line.strip().split(',')[-1])
    return flags


# Input Images,这个是MNIST数据集的数据数字信息
def get_images(filename, bol=False, length=10000):
    # Parameters -
    #  1. filename - FORMAT: filepath/filename
    #  2. bol - (default -False)-- get images for full length or not
    #  3. length of input images (default=10000)
    length = length*784
    with open(filename,'rb') as f:
        byte_=f.read()
        i = 16
        data = []
        while True:
            byte = byte_[i:i+1]
            if len(byte) == 0:
                break
            if i == length+16 and bol==False:
                break
            val = int.from_bytes(byte,byteorder='big', signed=False)
            data.append(val/255)
            i=i+1
    return data


# Input Lables 这是MNIST数据集的标签处理函数
def get_labels(filename):
    # Parameters -
    #  1. filename - FORMAT: filepath/filename
    with open(filename,'rb') as f:
        byte_=f.read()
        i = 8
        data = []
        while True:
            byte = byte_[i:i+1]
            if len(byte) == 0:
                break
            hexadecimal = binascii.hexlify(byte)
            decimal = int(hexadecimal, 16)
            data.append(decimal)
            i = i+1
    return data



if __name__=='__main__':
    #这部分的数据是训练数据
    train_data = get_images("D:\\project\\SVM-MNISTS\\train_data\\train-images.idx3-ubyte", length=60000) #win
    train_labels = get_labels('D:\\project\\SVM-MNISTS\\train_data\\train-labels.idx1-ubyte')  #win
    train_data = np.asmatrix(train_data[:(60000*784)]).reshape(60000, 784)
    #这部分的数据是训练数据
    test_data=get_images('D:\\project\\SVM-MNISTS\\test_data\\t10k-images.idx3-ubyte',True)  # True: for full length #win
    test_labels=get_labels('D:\\project\\SVM-MNISTS\\test_data\\t10k-labels.idx1-ubyte')  #win
    test_data = np.asmatrix(test_data).reshape(10000, 784)
    
    
    print("模型训练中......")#这部分模型是KNN模型
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(train_data, train_labels) 
    neigh.score(test_data, test_labels)
    
    #这部分是朴素贝叶斯的训练模型
    gnb = GaussianNB()
    clf = gnb.fit(train_data, train_labels)
    clf.score(test_data, test_labels)


    #这部分是决策树模型
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data, train_labels)
    clf.score(test_data, test_labels)
    
    
    #这部分是深度学习模型,神经网络模型
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    clf = clf.fit(train_data, train_labels)
    result=clf.score(test_data, test_labels)
    
    