# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 21:11:37 2018

@author: hello
"""

import binascii
from sklearn import svm
import pickle
import numpy as np
import time
import matplotlib.pyplot  as plt



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



def train(g):
    #train_data = get_images("D:\\project\\SVM-MNISTS\\train_data\\train-images.idx3-ubyte", length=60000) #win
    train_data=get_images('/Users/macbook/documents/project/SVM-MNISTS/train_data/train-images.idx3-ubyte', length=60000)  #mac
    #train_labels = get_labels('D:\\project\\SVM-MNISTS\\train_data\\train-labels.idx1-ubyte')  #win
    train_labels=get_labels('/Users/macbook/documents/project/SVM-MNISTS/train_data/train-labels.idx1-ubyte') #mac
   # clf = svm.SVC()
    #clf=svm.SVC(C=0.8,  gamma=20)
    clf=svm.SVC(C=100.0, kernel='rbf', gamma=0.4)
    #clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr') 
    #在这个地方使用
    train_data = np.asmatrix(train_data[:(60000*784)]).reshape(60000, 784)
    #train_data=datamats  #iris数据集测试结果
    #train_labels=flags
    
    print("模型训练中......")
    clf.fit(train_data, train_labels)#[:60000])
    print("模型训练完成......")
    # save the model to disk
    filename = 'D:\\project\\SVM-MNISTS\\finalized_model_50000_f.sav'
    filename = '/Users/macbook/documents/project/Machine-Learning/PCA/finalized_model_f.sav'
    pickle.dump(clf, open(filename, 'wb'))
    print("Succeed!")
    return filename


def test(filename):
   # filename = 'D:\\project\\SVM-MNIST\\finalized_model_50000_f.sav'

    # load the model from disk
    clf = pickle.load(open(filename, 'rb'))
    
   # test_data=get_images('D:\\project\\SVM-MNISTS\\test_data\\t10k-images.idx3-ubyte',True)  # True: for full length #win
    test_data=get_images('/Users/macbook/documents/project/SVM-MNISTS/test_data/t10k-images.idx3-ubyte',True)  #mac
   # test_labels=get_labels('D:\\project\\SVM-MNISTS\\test_data\\t10k-labels.idx1-ubyte')  #win
    test_labels=get_labels('/Users/macbook/documents/project/SVM-MNISTS/test_data/t10k-labels.idx1-ubyte')  #mac
    test_data = np.asmatrix(test_data).reshape(10000, 784)
    print("测试进行中......")
    #test_data=loadiris(p1)
   # test_labels=loadflags(p1)
    result = clf.score(test_data, test_labels)
    print("测试完成......")
    #print("Accuracy: ",result)
    return result


if __name__=="__main__": 
    g=np.linspace(0.1,1,10)
    g=g.tolist()
    results=[]
    for i in g:
        start = time.clock()
        filename=train(i)
        result=test(filename)
        print("训练的精确度是: ",result)
        results.append(result)
        end = time.clock()
        print (end-start)
    plt.scatter(g,results)
  
    
    
    
    