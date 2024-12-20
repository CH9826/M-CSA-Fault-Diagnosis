'''
A temporal convolutional network.



@author: malteschilling@googlemail.com
'''
import numpy as np
import os
import random
import re
import shutil
import time
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import keras
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus'] = False##用来正常显示负号

from matplotlib.ticker import MultipleLocator
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
#from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.metrics import confusion_matrix,auc
from keras.layers.normalization.layer_normalization import *
from keras.layers.normalization.batch_normalization import *

from keras.models import Sequential,Model##堆叠，通过堆叠许多层，构建出深度神经网络
from keras.layers import Dense, Flatten, Dropout, Input, add, concatenate, SpatialDropout1D#Flatten是展平层
from keras.layers import UpSampling1D
from keras.layers import LeakyReLU
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras import optimizers
from keras import *

from keras import backend as K##后端设置，可以选择不同的架构
from keras.layers import Layer

from keras.callbacks import TensorBoard
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize


# class LossHistory(keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.losses = []
 
#     def on_batch_end(self, batch, logs={}):
#         self.losses.append(logs.get('loss'))
# history = LossHistory()        
        
def plot_confusion_matrix(matrix, classes, savename):
    matrix = matrix.astype(np.float)
    linesum = matrix.sum(1)
    linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
    matrix /= linesum
    # plot
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    for i in range(matrix.shape[0]):
           ax.text(i, i, str('%.2f' % (matrix[i, i] * 100)), va='center', ha='center')
    ax.set_xticklabels([''] + classes, rotation=90)
    ax.set_yticklabels([''] + classes)
    #ax.set_xlabel('Predicted label')
    #ax.set_ylabel('True label')
    ax.set_xlabel('诊断的故障标签',color='w')
    ax.set_ylabel('真实的故障标签',color='w')
    plt.tick_params(axis='x',colors='w')
    plt.tick_params(axis='y',colors='w')
    #ax.patch.set_facecolor('greenyellow')
    fig.patch.set_facecolor('#004775')
    #plt.tick_params(axis='x',colors='w')
    #plt.tick_params(axis='y',colors='w')
    #plt.rcParams['figure.facecolor'] = 'b'
    plt.rcParams['savefig.dpi'] = 700 # 图片像素
    plt.rcParams['figure.dpi'] = 700 # 分辨率
    plt.savefig(savename, bbox_inches='tight' )

def huatu(Y_test1, result,classes):
    m=int(max(set(classes)))
    label_name=[]
    for i in range(1,m+1):
        num= str(i)
        str1='故障'
        name=str1+num
        label_name.append(name)
    plt.clf()
            #binary_sdae_result为模型测试输出(一列，类似于[0,1,3,2,1])，y_train为理想输出(一列)
    cm_CNNLSTM = confusion_matrix(result, Y_test1)
    plot_confusion_matrix(cm_CNNLSTM, label_name, '1.png')
    #plt.rcParams['figure.facecolor'] = 'b'
    plt.show()

def Plt_ROC(FPR, recall):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #plt.plot(FPR, recall, c='w', label='ROC curve')  # ROC 曲线
    plt.plot(FPR, recall, c='greenyellow') 
    plt.title('ROC',color='w')  # 设置标题
    plt.plot([0,1],[0,1],'r--')
    #plt.xlim([-0.05, 1.05])
    #plt.ylim([-0.05, 1.05])
    plt.xlabel('FPR',color='w')
    plt.ylabel('Recall',color='w')
    #plt.legend(loc='lower right')
    #A=color(0,71,117)
    ax.patch.set_facecolor('#004775')
    fig.patch.set_facecolor('#004775')
    axe = plt.gca() 
    axe.spines['bottom'].set_color('w')
    axe.spines['left'].set_color('w')
    axe.spines['top'].set_color('w')
    axe.spines['right'].set_color('w')
    #axe.spines['left'].set_color('w')
    plt.tick_params(axis='x',colors='w')
    plt.tick_params(axis='y',colors='w')
    #plt.rcParams['figure.facecolor'] = 'b'
    plt.show()
    plt.rcParams['savefig.dpi'] = 700 # 图片像素
    plt.rcParams['figure.dpi'] = 700 # 分辨率
    plt.savefig('ROC.png' ,bbox_inches='tight' )
    
def ConvolutionalAE (inputTensor):
    shortcut = inputTensor
    # ENCODER
    x = Conv1D(32, (3), activation=None, padding='same')(inputTensor)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling1D((2), padding='same')(x)
    x = Conv1D(32, (3), activation=None, padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling1D((2), padding='same')(x)
    x = Conv1D(64, (3), activation=None, padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    encoded = MaxPooling1D((2), padding='same')(x)

    x = Conv1D(64, (3), activation=None, padding='same')(encoded)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling1D((2))(x)
    x = Conv1D(32, (3), activation=None, padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling1D((2))(x) 
    x = Conv1D(32, (3), activation=None, padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling1D((2))(x)              ##   <-- change here (was 4)
    decoded = Conv1D(12, (3), activation=None, padding='same')(x)
    decoded = LeakyReLU(alpha=0.1)(decoded)
    output = concatenate([decoded, shortcut])
    return output    

def CNNBlock( inputTensor, filter_number, kernel_size=3, 
        padding='causal', dilation_rate=1):
    shortcut = inputTensor

    convLayer1 = Conv1D(filters=filter_number, kernel_size=kernel_size, padding=padding, dilation_rate=dilation_rate)#dilation_rate稀疏速率, activation='relu'
    conv_output = convLayer1(inputTensor)
    conv_output = LeakyReLU(alpha=0.1)(conv_output)
    conv_output = SpatialDropout1D(0.25)(conv_output)
    ##conv_output = MaxPooling1D(pool_size=2)(conv_output)
    convLayer2 = Conv1D(filters=filter_number, kernel_size=kernel_size, padding=padding, dilation_rate=dilation_rate)##, activation='relu'
    conv_output = convLayer2(conv_output)
    conv_output = LeakyReLU(alpha=0.1)(conv_output)    
    conv_output = SpatialDropout1D(0.25)(conv_output)

    # Residual connection:
#     multi_resolution_concat = concatenate([conv_output, shortcut])
#     output = Conv1D(filter_number, 1, padding='same')(multi_resolution_concat)
    if (inputTensor.shape[2] != conv_output.shape[2]):
        res_connection = Conv1D(filter_number, 1, padding='same')(inputTensor)
        output = add([conv_output, res_connection])
    else:
        output = add([conv_output, inputTensor])

    return output
def CNNBlock1( inputTensor, filter_number, kernel_size=3, 
        padding='causal', dilation_rate=1):
    shortcut = inputTensor
    convLayer1 = Conv1D(filters=filter_number, kernel_size=kernel_size, padding=padding, dilation_rate=dilation_rate)#dilation_rate稀疏速率, activation='relu'
    conv_output = convLayer1(inputTensor)
    conv_output = LeakyReLU(alpha=0.1)(conv_output)
    conv_output = SpatialDropout1D(0.25)(conv_output)
    ##conv_output = MaxPooling1D(pool_size=2)(conv_output)
    convLayer2 = Conv1D(filters=filter_number, kernel_size=kernel_size, padding=padding, dilation_rate=dilation_rate)##, activation='relu'
    conv_output = convLayer2(conv_output)
    conv_output = LeakyReLU(alpha=0.1)(conv_output)    
    conv_output = SpatialDropout1D(0.25)(conv_output)

    # Residual connection:
#     multi_resolution_concat = concatenate([conv_output, shortcut])
#     output = Conv1D(filter_number, 1, padding='same')(multi_resolution_concat)
    if (shortcut.shape[2] != conv_output.shape[2]):
        res_connection = Conv1D(filter_number, 1, padding='same')(shortcut)
        output = concatenate([conv_output, res_connection])
    else:
        output = concatenate([conv_output, inputTensor])

    return output
def CNNBlock2( inputTensor, filter_number, kernel_size=3, 
        padding='causal', dilation_rate=1):
    shortcut = inputTensor

    convLayer1 = Conv1D(filters=filter_number, kernel_size=kernel_size, padding=padding, dilation_rate=dilation_rate)#dilation_rate稀疏速率
    conv_output = convLayer1(inputTensor)
    conv_output = LeakyReLU(alpha=0.1)(conv_output)
    conv_output = SpatialDropout1D(0.25)(conv_output)

    
    convLayer2 = Conv1D(filters=filter_number, kernel_size=kernel_size, padding=padding, dilation_rate=dilation_rate )
    conv_output = convLayer2(conv_output)
    conv_output = LeakyReLU(alpha=0.1)(conv_output)
    conv_output = SpatialDropout1D(0.25)(conv_output)
    

    # Residual connection:
#     multi_resolution_concat = concatenate([conv_output, shortcut])
#     output = Conv1D(filter_number, 1, padding='same')(multi_resolution_concat)
    if (shortcut.shape[2] != conv_output.shape[2]):
        res_connection = Conv1D(filter_number, 1, padding='same')(shortcut)
        output = concatenate([conv_output, res_connection])
    else:
        output = concatenate([conv_output, shortcut])
    #print(output.shape)
    output=output[:,output.shape[1]-1,:]
    #print(output.shape)
    #output=np.reshape(output,(output.shape[0],output.shape[2]))
    return output
class CNN:

    def __init__ (self, input_shape, output_number, 
            module_layers=3, 
            filter_number=64, kernel_size=3, padding='causal', 
            regression=False,
            epochs=10):
        self._epochs = epochs
        self._batch_size = 256
        
        # Setup Model 
        print(input_shape)
        print(output_number)
        inputTensor = Input(shape=(input_shape[0], input_shape[1]))
#         caeOutput = ConvolutionalAE ( inputTensor )
        #tcnOutput = TemporalConvolutionalBlock1( inputTensor, filter_number=filter_number, kernel_size=kernel_size,
        #       padding=padding, dilation_rate=1 )        
        Output = CNNBlock1( inputTensor, filter_number=filter_number, kernel_size=kernel_size,
                padding=padding, dilation_rate=1 )
#         Output = CNNBlock1( Output, filter_number=filter_number, kernel_size=kernel_size,
#                 padding=padding, dilation_rate=2 )
#         Output = CNNBlock1( Output, filter_number=filter_number, kernel_size=kernel_size,
#                 padding=padding, dilation_rate=4 )
        Output = CNNBlock2( Output, filter_number=filter_number, kernel_size=kernel_size,
                padding=padding, dilation_rate=1 )
        #tcnOutput = TemporalConvolutionalBlock( tcnOutput, filter_number=filter_number, kernel_size=kernel_size,
        #        padding=padding, dilation_rate=4 )
        #print(tcnOutput)
        self.regression = regression
        if self.regression:
            outputLayer = Dense(output_number, activation=None)
            finalOutput = outputLayer(Output)
            finalOutput = LeakyReLU(alpha=0.1)(finalOutput)
            self.model = Model(inputTensor,finalOutput)##keras中的model模块，通用模型能够比较灵活地构造网络结构，设定各层级的关系。
            #adam = optimizers.Adam(lr=0.002, clipnorm=1.)
            adam = tf.keras.optimizers.Adam(lr=0.002, clipnorm=1.)
            self.model.compile(adam, loss='mean_squared_error')##编译模型
        else:
            
            outputLayer = Dense(output_number,activation='softmax')
            #outputLayer=round(outputLayer)
            finalOutput = outputLayer(Output)
            self.model = Model(inputTensor,finalOutput)
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        #self.model.summary()
        print("Size of receptive field: ", self.calculate_receptive_field_size() )

    def calculate_receptive_field_size(self):
        rec_field_size = 1.
        for layer in self.model.layers:
            if isinstance(layer, Conv1D):
                # Calculate size of receptive field:
                # rec_#(n) = rec_#(n-1) + 2 * [kernel_size(n)-1] * dilation(n).
                rec_field_size += (layer.kernel_size[0]-1) * (layer.dilation_rate[0])
        return rec_field_size
   
    def train_model(self, train_X, train_y, sample_y_flatten=None, val_X=None, val_y=None, validation_split=0.3, epochs=None):    
        epochs = self._epochs if epochs is None else epochs
        
        import time
        now = time.strftime("%c")
        # Train the network on the training data.
        if (val_X is None):
            # Train the network on the training data.
            history=self.model.fit(train_X, train_y, validation_split=validation_split, 
                epochs=epochs, batch_size=self._batch_size, verbose=1)
            history.history.keys()  # 查看字典的键
            loss = history.history['loss']  # 测试集损失
            acc = history.history['accuracy']  # 测试集准确率
            val_loss = history.history['val_loss']  # 验证集损失
            val_acc = history.history['val_accuracy']  # 验证集准确率
#             print(loss)
#             print(acc)
#             print(val_loss)
#             print(val_acc)

            return self.model,loss,acc,val_loss,val_acc
            return 
            ## verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
            ##callbacks=[TensorBoard(log_dir='./logdir/block/'+now)] 
        else:
            # Train the network on the training data.指定验证集数据
            Hist = self.model.fit(train_X, train_y, validation_data=(val_X, val_y), 
                epochs=epochs, batch_size=self._batch_size, verbose=1)
            train_loss = Hist.history['loss']
            test_loss = Hist.history['val_loss']##这是list,需要转化为np.array
            train_loss = np.array(train_loss)
            test_loss = np.array(test_loss)
            df3 = pd.DataFrame(train_loss)
            df4 = pd.DataFrame(test_loss)
            df_sum1 = pd.concat([df3,df4],axis=1)
            df_sum1.to_csv("loss_results.csv",index = False,sep = ',')
            #print(train_loss, test_loss)
            predict_y = self.model.predict(val_X)
            #predict_y = predict_y[:,39,:]
            predict_y_flatten=np.reshape(predict_y,[-1])
            #print(predict_y_flatten, predict_y.shape, predict_y_flatten.shape)
            df1 = pd.DataFrame(predict_y_flatten)#;sample_truth_np;sample_loss_np   
            df2 = pd.DataFrame(sample_y_flatten)
            #df3 = pd.DataFrame(sample_loss)
            df_sum = pd.concat([df1,df2],axis=1)
            df_sum.to_csv("RUL_results.csv",index = False,sep = ',')
            #存入不同的测试集评价结果
            model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # 回归评估指标对象集
            model_metrics_list = []  # 回归评估指标列表
            tmp_list = []  # 每个内循环的临时结果列表
            for m in model_metrics_name:  # 循环每个指标对象
                tmp_score = m(predict_y_flatten, sample_y_flatten)  # 计算每个回归指标结果
                tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
            model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表
            df5 = pd.DataFrame(model_metrics_list)  # 建立回归指标的数据框
            df5.to_csv("metrics_test_loss.csv",index = False,sep = ',')
            
    def y_score(self,val_X,n_fenlei):
        y_score_=self.model.predict(val_X)
        if n_fenlei==2:
            y_score = y_score_[:,1]
        else:
            y_score = y_score_.ravel()
        return y_score
    
    def predict_model(self,val_X):
        import time
        now = time.strftime("%c")
       
       
        predict_y = self.model.predict(val_X)
        predict_y1 =np.argmax(predict_y,axis=1)+1##解码
        #predict_y = predict_y[:,39,:]
        predict_y_flatten=np.reshape(predict_y1,[-1])
        
        #print(predict_y_flatten, predict_y.shape, predict_y_flatten.shape)
        return predict_y_flatten,predict_y
        
        #df1 = pd.DataFrame(predict_y_flatten)#;sample_truth_np;sample_loss_np   
        #df2 = pd.DataFrame(Y_test1)
        ###df3 = pd.DataFrame(sample_loss)
        #df_sum = pd.concat([df1,df2],axis=1)
        #df_sum.to_csv("results.csv",index = False,sep = ',')
        ###存入不同的测试集评价结果
        #model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # 回归评估指标对象集
        #model_metrics_list = []  # 回归评估指标列表
        #tmp_list = []  # 每个内循环的临时结果列表
        #for m in model_metrics_name:  # 循环每个指标对象
        #    tmp_score = m(predict_y_flatten, Y_test1)  # 计算每个回归指标结果
        #    tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
        #model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表
        #df5 = pd.DataFrame(model_metrics_list)  # 建立回归指标的数据框
        #df5.to_csv("metrics_test_loss.csv",index = False,sep = ',')        
            #结束
        ##callbacks=[TensorBoard(log_dir='./logdir/block/'+now)] 
        #print(self.model.predict(train_X[100:110]))
        #print(train_y[100:110])
        #print(self.model.predict(val_X[200:210]))
        #print(val_y[200:210])
        
#     def show_accuracy_over_time(self, set_X, targets_y):
#         import numpy as np
#         set_y = self.model.predict(set_X)
#         #targets_y_ext = np.tile(targets_y, (set_X.shape[1],1))
#         miss_classification_over_time = 1.* np.sum( 
#             np.argmax(set_y, axis=2) <> np.argmax(targets_y,axis=2) , axis=0)
#         accuracy_over_time = 1. - miss_classification_over_time/set_X.shape[0]
#         import matplotlib.pyplot as plt
#         self.fig = plt.figure(figsize=(8, 6))    
#         ax_accuracy = plt.subplot(111) 
#         ax_accuracy.plot(accuracy_over_time)
#         plt.show()
#         
#     def show_loss_over_time(self, set_X, targets_y):
#         import numpy as np
#         set_y = self.model.predict(set_X)
#         loss_over_time = np.mean(np.square(set_y - targets_y), axis=(0,2))
#         print("Minimal loss over time: ", np.min(loss_over_time), 
#             " - at time: ", np.argmin(loss_over_time))
#         import matplotlib.pyplot as plt
#         self.fig = plt.figure(figsize=(8, 6))    
#         ax_loss = plt.subplot(111) 
#         ax_loss.plot(loss_over_time)
#         plt.show()
        
#    def get_prediction_without_padding(self, set_X):
 #       complete_time_series = self.model.predict(set_X)
  #      return complete_time_series[:,-1,:]
        
#    def get_targets_without_padding(self, set_y):
 #       return set_y[:,-1,:]