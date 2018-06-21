from input_data import get_labels,get_images
from sklearn import svm
import pickle
import numpy as np

train_data = get_images("D:\\project\\SVM-MNIST\\train_data\\train-images.idx3-ubyte", length=60000)
train_labels = get_labels('D:\\project\\SVM-MNIST\\train_data\\train-labels.idx1-ubyte')

clf = svm.SVC()
train_data = np.asmatrix(train_data[:(60000*784)]).reshape(60000, 784)

clf.fit(train_data, train_labels[:60000])

# save the model to disk
filename = 'D:\\project\\SVM-MNIST\\finalized_model_50000_f.sav'
pickle.dump(clf, open(filename, 'wb'))
print("Succeed!")

