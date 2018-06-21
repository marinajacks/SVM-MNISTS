# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 21:11:37 2018

@author: hello
"""

import binascii
from sklearn import svm
import pickle
import numpy as np



# Input Images
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


# Input Lables
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



def train():
    train_data = get_images("D:\\project\\SVM-MNIST\\train_data\\train-images.idx3-ubyte", length=60000)
    train_labels = get_labels('D:\\project\\SVM-MNIST\\train_data\\train-labels.idx1-ubyte')
    
    clf = svm.SVC()
    train_data = np.asmatrix(train_data[:(60000*784)]).reshape(60000, 784)
    
    print("模型训练中......")
    clf.fit(train_data, train_labels[:60000])
    print("模型训练完成......")
    # save the model to disk
    filename = 'D:\\project\\SVM-MNIST\\finalized_model_50000_f.sav'
    pickle.dump(clf, open(filename, 'wb'))
    print("Succeed!")
    return filename


def test(filename):
   # filename = 'D:\\project\\SVM-MNIST\\finalized_model_50000_f.sav'

    # load the model from disk
    clf = pickle.load(open(filename, 'rb'))
    
    test_data=get_images('D:\\project\\SVM-MNIST\\test_data\\t10k-images.idx3-ubyte',True)  # True: for full length
    test_labels=get_labels('D:\\project\\SVM-MNIST\\test_data\\t10k-labels.idx1-ubyte')
    
    test_data = np.asmatrix(test_data).reshape(10000, 784)
    print("测试进行中......")
    result = clf.score(test_data, test_labels)
    print("测试完成......")
    #print("Accuracy: ",result)
    return result


if __name__=="__main__":
    filename=train()
    result=test(filename)
    print("训练的精确度是: ",100*result+'%')
    
    
    
    