#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import argparse##从黑窗口命令行直接读取参数
import numpy as np
import os
import pandas as pd
import random
import time
import matplotlib.pyplot as plt  

from sklearn.manifold import TSNE 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from model.TemporalConvolutionalNetwork_ import*
from model.data_model_RUL1 import TrainDataSet,TestDataSet
import h5py
from keras.models import load_model



# In[3]:


#num_epochs = 40
def Train_TCN(DataFrame1,path_save_model,num_epochs,kernel_size,filter_number,num_steps):
    RUL_Data=TrainDataSet(
                          train_DataFrame = DataFrame1,
                          #test_DataFrame = DataFrame2,
                          num_steps=num_steps,
                          )
    dataset_RUL=RUL_Data
    #print(dataset_RUL.train_X)
    
    X_train = dataset_RUL.train_X
    Y_train = dataset_RUL.train_y
    #x_test = dataset_RUL.test_X_list
    #y_test = dataset_RUL.test_y_list
    #x_test = dataset_RUL.test_X
            #y_test = dataset_RUL.test_y
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    Y_train = Y_train[:,num_steps-1,:]
    X_train, X_test, Y_train, Y_test1 = train_test_split(X_train,Y_train, test_size=0.3, random_state=0)
    regression = False
            
    encoder = LabelEncoder()
    encoded_Y = encoder.fit_transform(Y_train)#将离散型的数据转换成 0 到 n−1 之间的数
    Y_train = np_utils.to_categorical(encoded_Y)#在二分类的时候直接转换为（0，1）
    
    
    Y_test1=Y_test1.flatten()
    print(X_train.shape[1],X_train.shape[2])
    
    start_time = time.time()  # 记录开始时间
    nn_model = TemporalConvolutionalNetwork(input_shape=(X_train.shape[1],X_train.shape[2]), 
                output_number=Y_train.shape[1],
                kernel_size=kernel_size, filter_number=filter_number, padding='causal',
                regression=regression) ##(train_X.shape[1],train_X.shape[2])TemporalConvolutionalNetwork因为数据是三维的，因为0维是索引
    model,loss,acc,val_loss,val_acc=nn_model.train_model(X_train, Y_train, sample_y_flatten=None,epochs=num_epochs)## 
    model.save(path_save_model)
    end_time = time.time()  # 记录结束时间
    training_duration = end_time - start_time  
    print(f"Training duration: {training_duration:.2f} seconds")
    
    
    result,token = nn_model.predict_model(X_test)
            
    classes_=sorted(list(set(Y_test1)))
    n_fenlei = len(classes_)
            
    huatu(Y_test1,result,classes_)
            
    y_test_all = label_binarize(Y_test1, classes=classes_)
    y_score=nn_model.y_score(X_test,n_fenlei=n_fenlei)
            #print(y_score_)
    FPR, recall, thresholds = roc_curve(y_true=y_test_all.ravel(),  # 真实标签是
                                y_score=y_score # 置信度，也可以是概率值
                                    )
    recall[0]=0
    Plt_ROC(FPR, recall)
    tsne = TSNE(n_components=2, random_state=0)  
    X_tsne = tsne.fit_transform(token.reshape(token.shape[0], -1))  # 将X_test展平后降维  
      
    plt.figure(figsize=(8, 6))  
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y_test1)  
    plt.colorbar(scatter)  
    plt.title('t-SNE visualization of test data')  
    plt.savefig('tsne_plot_tcn.png')  
    plt.show()
    
    return loss,acc,val_loss,val_acc
#     print(loss)
#     print(acc)
#     print(val_loss)
#     print(val_acc)
    
def Test_TCN(DataFrame2,path_load_model,num_steps):  
            RUL_Data=TestDataSet(
                          #train_DataFrame = DataFrame1,
                          test_DataFrame = DataFrame2,
                          num_steps=num_steps,
                          )
            dataset_RUL=RUL_Data
            test_X = dataset_RUL.test_X
            test_X = np.array(test_X)
            #test_y = np.array(test_y)
            clf2 = load_model(path_load_model)
            predict_y = clf2.predict(test_X)
            print(predict_y.shape)
            predict_y =np.argmax(predict_y,axis=1)+1##解码
            Identification_result=np.reshape(predict_y,[-1])
            number=len(set(Identification_result))
            return Identification_result,number
            


# In[ ]:




