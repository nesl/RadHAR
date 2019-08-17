"""
SVM Classifier on the flattened VOXELS after PCA

Usage:

- extract_path is the where the extracted data samples are available.
- checkpoint_model_path is the path where to checkpoint the trained models during the training process


EXAMPLE: SPECIFICATION

extract_path = '/Users/sandeep/Research/Ti-mmWave/data/extract/Train_Data_voxels_'
checkpoint_model_path="/Users/sandeep/Research/Ti-mmWave/data/extract/LSTM"
"""

extract_path = '/Users/sandeep/Research/Ti-mmWave/data/extract/Train_Data_voxels_'
checkpoint_model_path="/Users/sandeep/Research/Ti-mmWave/data/extract/SVM"



import glob
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pickle



sub_dirs=['boxing','jack','jump','squats','walk']


frame_tog = [60]


#loading the train data
Data_path = extract_path+'boxing'

data = np.load(Data_path+'.npz')
train_data = data['arr_0']
train_data = np.array(train_data,dtype=np.dtype(np.int32))
train_label = data['arr_1']

del data
#print(train_data.shape,train_label.shape)

Data_path = extract_path+'jack'
data = np.load(Data_path+'.npz')
train_data = np.concatenate((train_data, data['arr_0']), axis=0)
train_label = np.concatenate((train_label, data['arr_1']), axis=0)


del data

#print(train_data.shape,train_label.shape)


Data_path = extract_path+'jump'
data = np.load(Data_path+'.npz')
train_data = np.concatenate((train_data, data['arr_0']), axis=0)
train_label = np.concatenate((train_label, data['arr_1']), axis=0)

del data
#print(train_data.shape,train_label.shape)

Data_path = extract_path+'squats'
data = np.load(Data_path+'.npz')
train_data = np.concatenate((train_data, data['arr_0']), axis=0)
train_label = np.concatenate((train_label, data['arr_1']), axis=0)

del data
#print(train_data.shape,train_label.shape)

Data_path = extract_path+'walk'
data = np.load(Data_path+'.npz')
train_data = np.concatenate((train_data, data['arr_0']), axis=0)
train_label = np.concatenate((train_label, data['arr_1']), axis=0)

del data

train_data = train_data.reshape(train_data.shape[0],train_data.shape[1], train_data.shape[2]*train_data.shape[3]*train_data.shape[4])


def number_encoding(y_data, sub_dirs, categories=5):
    Mapping=dict()

    count=1
    for i in sub_dirs:
        Mapping[i]=count
        count=count+1

    y_features2=[]
    for i in range(len(y_data)):
        Type=y_data[i]
        #print(Type)
        lab=Mapping[Type]
        y_features2.append(lab)

    y_features=np.array(y_features2)
    return y_features

labels2 = number_encoding(train_label, sub_dirs, categories=5)
train_label = labels2
train_data = train_data.reshape(train_data.shape[0],train_data.shape[1]*train_data.shape[2])


print('Training Data Shape is:')
print(train_data.shape,train_label.shape)


feat_train, feat_test, lbl_train, lbl_test = train_test_split(train_data, train_label, test_size=0.0001, random_state=42)
del train_data, train_label

#PCA is applied with every component so the variance ratio can be graphed.
pca = PCA(n_components=6000, svd_solver='randomized', whiten=True).fit(feat_train)
pca_feat_train = pca.transform(feat_train)
print(pca_feat_train.shape,lbl_train.shape)


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


param_grid = {'C': [1e3],
              'gamma': [ 0.0001], }

svm = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
svm = svm.fit(pca_feat_train, lbl_train)

print('Best estimator: ')
print(svm.best_estimator_)



## Saving the SVM model

# filehandler = open(checkpoint_model_path, 'wb')
# pickle.dump(svm, filehandler)
