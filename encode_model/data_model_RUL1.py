# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 16:45:29 2018

@author: Hang Wang
"""
# In[]
import numpy as np
import os
import pandas as pd
import random
import time
from sklearn import preprocessing

# In[]
random.seed(time.time())


class TrainDataSet(object):
    def __init__(self,
                 train_DataFrame,
                 num_steps=10,
                 ):

        self.num_steps = num_steps
        #print("num_steps:",self.num_steps)
        #self.test_ratio = test_ratio
        #unit_DataFrame=pd.read_csv(scaled_train_path,header=None,encoding='utf-8')

        # In[]
        unit_number_RUL_scaled_list=self._read_train_data(train_DataFrame)#读取归一化的传感器数据以及剩余寿命数据
        #unit_number_test_scaled_list=self._read_test_data(test_DataFrame)#读取归一化的传感器数据以及剩余寿命数据
        # In[]
       
        # In[]
        self.train_X_list,self.train_y_list=self._generate_train_from_unit_list(
                num_steps,
                unit_number_RUL_scaled_list)
        #self.final_test_X_list=self._generate_final_test_from_unit_list(#读取真正的测试集
        #        num_steps,
        #        unit_number_test_scaled_list)
        #将train_X_list和train_y_list中的元素分别垂直拼接
        train_X_tmp=self.train_X_list[0]
        train_y_tmp=self.train_y_list[0]
        for i in range(1,len(self.train_X_list)):
            train_X_tmp=np.vstack((train_X_tmp,self.train_X_list[i]))
            train_y_tmp=np.vstack((train_y_tmp,self.train_y_list[i]))
        '''
        从train excel中制作用于训练的数据，用一个大小为num_steps的
        滑动窗口来获取每个用于训练的数据块，因此数据之间是有重叠的
        '''
        self.train_X=train_X_tmp
        self.train_y=train_y_tmp
        # In[]

       #self.test_X_list,self.test_y_list=self._generate_test_from_unit_list(
       #        num_steps,
       #        unit_number_test_scaled_list)
       #
       #test_X_tmp=self.test_X_list[0]
       ##test_y_tmp=self.final_test_X_list[0]
       #for i in range(1,len(self.test_X_list)):
       #    test_X_tmp=np.vstack((test_X_tmp,self.test_X_list[i]))
       #    #test_y_tmp=np.vstack((test_y_tmp,self.test_y_list[i]))
       #    
       #self.test_X=test_X_tmp
        #self.test_y=test_y_tmp

        # In[]
        '''
        将train_X进一步随机划分为训练集和测试集，以供备用
      
        train_indices=list(range(len(self.train_X)))
        random.shuffle(train_indices) ####（随机打乱0,1,2,...len-1）
        # In[]
        test_ratio=0.1
        self.training_X=self.train_X[train_indices[0:int(len(train_indices)*(1-test_ratio))]]
        self.training_y=self.train_y[train_indices[0:int(len(train_indices)*(1-test_ratio))]]
        self.testing_X=self.train_X[train_indices[int(len(train_indices)*(1-test_ratio)):]]
        self.testing_y=self.train_y[train_indices[int(len(train_indices)*(1-test_ratio)):]]
        '''


    # In[]
    def _read_train_data(self,DataFrame,isCut=True):
        """
        适用于训练集和测试集，归一化和未归一化
        根据给定的路径，读取其中多台发动机的运行数据，存在一个list中，
        list的每个元素是一台发动机的运行数据（二维矩阵）
        """

        #unit_DataFrame=pd.read_excel(path,header=None,encoding='utf-8')#engine='openpyxl'
        #print(unit_DataFrame)
        unit_np=DataFrame.values  #将数据集放在一个NumPy array中
        unit_number_redundant=unit_np[:,-1]  #提取出冗余的unit编号
        unit_number=np.unique(unit_number_redundant)  #删除unit编号中的冗余部分
        unit_nums=unit_number.shape[0]  #发动机编号数
        #将每台发动机的运行数据存在一个二维列表中，将所有的二维列表存在一个list中
        unit_number_list=[]
        for i in range(0,unit_nums):
            condition_i=unit_np[:,-1]==i+1#找出对应编号的数据下标集合
            unit_index_i=np.where(condition_i)
            unit_number_i_index=unit_index_i[0]
            unit_number_i=unit_np[unit_number_i_index,:]
            unit_number_i_RULlabel=unit_number_i[:,unit_number_i.shape[1]-1]##20190904
            #print(unit_number_i[:,unit_number_i.shape[1]-2:unit_number_i.shape[1]-1])##20190904
            if(isCut):
                unit_number_i_scale=preprocessing.scale(unit_number_i[:,0:unit_number_i.shape[1]-1])##20190903此处注意，有冒号就表示不包含最后一列或行            
#             print(unit_number_i_scale[:,unit_number_i_scale.shape[1]-1])##20190904
#             print(unit_number_i_scale.shape[1])##20190904
            unit_number_i = np.column_stack((unit_number_i_scale,unit_number_i_RULlabel))##此处两个括号20190904,hstack不能针对稀疏矩
            #print(unit_number_i.shape[1])##20190904
            unit_number_list.append(unit_number_i)##20190904
        return unit_number_list
    
    def _generate_train_from_one_unit(self,multi_seq,TIMESTEPS=10):
        X = []
        Y = []
        # 序列的第i项和后面的TIMESTEPS-1项合在一起作为输入;
        # 第i+TIMESTEPS项和后面的PREDICT_STEPS-1项作为输出
        # 即用数据的前TIMESTPES个点的信息，预测后面的PREDICT_STEPS个点的值
        for i in range(len(multi_seq) - TIMESTEPS):
            X.append(multi_seq[i:i + TIMESTEPS,0:multi_seq.shape[1]-1])###正确####20190904
            Y.append([multi_seq[i:i + TIMESTEPS,multi_seq.shape[1]-1]])
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
    def _generate_train_from_unit_list(self,num_steps,unit_number_RUL_scaled_list):
        '''
        从train excel中制作用于训练的数据，用一个大小为num_steps的
        滑动窗口来获取每个用于训练的数据块，因此数据之间是有重叠的
        '''
        # In[]
        train_X_list=[]
        train_Y_list=[]
        for i in range(len(unit_number_RUL_scaled_list)):
            # In
            unit_number_i=unit_number_RUL_scaled_list[i]#取出第i台发动机的数据
            unit_number_i_var=unit_number_i.var(axis=0)#计算各传感器的方差
            # In
            good_index_i=unit_number_i_var>-1
            unit_number_i_good=unit_number_i[:,good_index_i]
            #print(unit_number_i_good.shape[0])##20190904
            # In

            # In
            #unit_number_i_good=unit_number_i_good[knee_point_i:unit_number_i_good.shape[0],:]
            unit_number_i_good=unit_number_i_good[0:unit_number_i_good.shape[0],:]
            #print(unit_number_i_good.shape[1])##20190904
            # In
            train_X_i=[]
            train_Y_i=[]
            train_X_i,train_Y_i=self._generate_train_from_one_unit(unit_number_i_good,TIMESTEPS=self.num_steps)
            #print("num_steps:",self.num_steps)
            #print("train_Y_i.shape:",train_Y_i.shape, i)
            train_Y_i_tmp=np.transpose(train_Y_i,[0,2,1])
            #print("train_Y_i_tmp.shape:",train_Y_i_tmp.shape)
            train_X_list.append(train_X_i)
            #train_X_list=preprocessing.scale(train_X_list)##20190903
            train_Y_list.append(train_Y_i_tmp)
            #train_Y_list_np = np.array(train_Y_list)
            #print("train_Y_list_np:",train_Y_list_np.shape)
        return train_X_list,train_Y_list
    
class TestDataSet(object):
    def __init__(self,
                 test_DataFrame,
                 num_steps,
                 ):
        self.num_steps = num_steps
        
        unit_number_test_scaled_list=self._read_test_data(test_DataFrame)#读取归一化的传感器数据以及剩余寿命数据
        
        self.test_X_list=self._generate_final_test_from_unit_list(
                num_steps,
                unit_number_test_scaled_list)
       
        test_X_tmp=self.test_X_list[0]
       ##test_y_tmp=self.final_test_X_list[0]
        for i in range(1,len(self.test_X_list)):
             test_X_tmp=np.vstack((test_X_tmp,self.test_X_list[i]))
       #    #test_y_tmp=np.vstack((test_y_tmp,self.test_y_list[i]))
       #    
        self.test_X=test_X_tmp
        #self.test_y=test_y_tmp

    
    def _read_test_data(self,DataFrame,isCut=True):
        """
        适用于训练集和测试集，归一化和未归一化
        根据给定的路径，读取其中多台发动机的运行数据，存在一个list中，
        list的每个元素是一台发动机的运行数据（二维矩阵）
        """

        #unit_DataFrame=pd.read_excel(path,header=None,encoding='utf-8')#engine='openpyxl'
        #print(unit_DataFrame)
        unit_np=DataFrame.values  #将数据集放在一个NumPy array中
        #print(unit_np)
        #print(unit_np[:,0])
        
        last = np.repeat([unit_np[-1]],repeats= 30 ,axis=0)
        unit_np = np.vstack([unit_np,last]) 
       
        #将每台发动机的运行数据存在一个二维列表中，将所有的二维列表存在一个list中
        unit_number_list=[]
        unit_number_i=unit_np
        #condition_i=unit_np[:,0]#找出对应编号的数据下标集合
       #    #print(condition_i)
       #    #print(condition_i.shape)
        #unit_index_i=np.where(condition_i)
        #unit_number_i_index=unit_index_i[0]
        #unit_number_i=unit_np[unit_number_i_index,:]
        #unit_number_i_RULlabel=unit_number_i[:,unit_number_i.shape[1]-1]##20190904
       #    #print(unit_number_i[:,unit_number_i.shape[1]-2:unit_number_i.shape[1]-1])##20190904
       #    #print(unit_number_i.shape)
        if(isCut):
            unit_number_i_scale=preprocessing.scale(unit_number_i[:,0:unit_number_i.shape[1]])
                #unit_number_i_scale=preprocessing.scale(unit_number_i[:,5:unit_number_i.shape[1]-1])##20190903此处注意，有冒号就表示不包含最后一列或行          
            #unit_number_i_scale=unit_number_i[:,0:unit_number_i.shape[1]-1]
            #print(unit_number_i_scale[:,unit_number_i_scale.shape[1]-1])##20190904
            #print(unit_number_i_scale.shape[1])##20190904
        #unit_number_i = np.column_stack((unit_number_i_scale,unit_number_i_RULlabel))##此处两个括号20190904,hstack不能针对稀疏矩
            #print(unit_number_i.shape[1])##20190904
        unit_number_list.append(unit_number_i_scale)##20190904
        #print(unit_number_list)
        return unit_number_list
    # In[]
    '''
    def _generate_test(self,test_scaled_path,test_RUL_path,isCut=True):
        test_unit_number_list=read_scaled_test_data(scaled_test_path,isCut=True)
        # In[]
        f=open('RUL_FD001.txt')
        raw_txt=f.read()
        print(raw_txt)
        f.close()
        str_list=raw_txt.split('\n')
        #str_list_last=str_list[100]
        #if(str_list_last==''):
            #print("bingo")
        test_RUL_list=[]
        for i in range(len(str_list)):
            if (str_list[i]!=''):
                test_RUL_list.append(int(str_list[i]))
                '''
    # In[]

    def _generate_test_from_one_unit(self,multi_seq,TIMESTEPS=30):
        X = []
        Y = []
      
        # 序列的第i项和后面的TIMESTEPS-1项合在一起作为输入;
        # 第i+TIMESTEPS项和后面的PREDICT_STEPS-1项作为输出
        # 即用数据的前TIMESTPES个点的信息，预测后面的PREDICT_STEPS个点的值
        for i in range(len(multi_seq) - TIMESTEPS):
            X.append(multi_seq[i:i + TIMESTEPS,0:multi_seq.shape[1]])###正确####20190904
            Y.append([multi_seq[i:i + TIMESTEPS,multi_seq.shape[1]-1]])
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
    
        #num_blocks=len(multi_seq)//TIMESTEPS  ###num_blocks=100//10=10
        #for i in range(len(multi_seq)//TIMESTEPS):
        #    X.append(multi_seq[len(multi_seq)-(num_blocks-i)*TIMESTEPS:len(multi_seq)-(num_blocks-i-1)*TIMESTEPS,0:multi_seq.shape[1]-1])###20190904
        #    Y.append([multi_seq[len(multi_seq)-(num_blocks-i)*TIMESTEPS:len(multi_seq)-(num_blocks-i-1)*TIMESTEPS,multi_seq.shape[1]-1]])
        #return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
    def _generate_test_from_unit_list(self,num_steps,unit_number_test_scaled_list):
        # In[]
        test_X_list=[]
        test_Y_list=[]
        for i in range(len(unit_number_test_scaled_list)):
            # In
            unit_number_i=unit_number_test_scaled_list[i]#取出第i台发动机的数据
            unit_number_i_var=unit_number_i.var(axis=0)#计算各传感器的方差
            # In
            good_index_i=unit_number_i_var>-1
            unit_number_i_good=unit_number_i[:,good_index_i] ###选取方差大于-1的传感器数据。第5列以后。
            # In
            # In
            #unit_number_i_good=unit_number_i_good[knee_point_i:unit_number_i_good.shape[0],:]
            unit_number_i_good=unit_number_i_good[0:unit_number_i_good.shape[0],:]#没有考虑拐点因素，选取了所有的数据
            # In
            test_X_i=[]
            test_Y_i=[]
            test_X_i,test_Y_i=self._generate_test_from_one_unit(unit_number_i_good,TIMESTEPS=self.num_steps)
            test_Y_i=np.transpose(test_Y_i,[0,2,1])
            test_X_list.append(test_X_i)
            #test_X_list=preprocessing.scale(test_X_list)##20190903
            test_Y_list.append(test_Y_i)
        return test_X_list,test_Y_list
    # In[]
    # In[]
    def _generate_final_test_from_one_unit(self,multi_seq,TIMESTEPS=10):
        X = []

        # 序列的第i项和后面的TIMESTEPS-1项合在一起作为输入;
        # 第i+TIMESTEPS项和后面的PREDICT_STEPS-1项作为输出
        # 即用数据的前TIMESTPES个点的信息，预测后面的PREDICT_STEPS个点的值
        for i in range(len(multi_seq) - TIMESTEPS):
            X.append(multi_seq[i:i + TIMESTEPS,0:multi_seq.shape[1]])##20190904

        return np.array(X, dtype=np.float32)
    def _generate_final_test_from_unit_list(self,num_steps,unit_number_test_scaled_list):
        '''
        从train excel中制作用于训练的数据，用一个大小为num_steps的
        滑动窗口来获取每个用于训练的数据块，因此数据之间是有重叠的
        '''
        # In[]
        train_X_list=[]
        for i in range(len(unit_number_test_scaled_list)):
            # In
            unit_number_i=unit_number_test_scaled_list[i]#取出第i台发动机的数据
            unit_number_i_var=unit_number_i.var(axis=0)#计算各传感器的方差
            # In
            good_index_i=unit_number_i_var>-1
            unit_number_i_good=unit_number_i[:,good_index_i]
            # In

            # In
            #unit_number_i_good=unit_number_i_good[knee_point_i:unit_number_i_good.shape[0],:]
            unit_number_i_good=unit_number_i_good[0:unit_number_i_good.shape[0],:]    ###第五列到最后一列数据，12维特征参数。
            # In
            train_X_i=[]

            train_X_i=self._generate_final_test_from_one_unit(unit_number_i_good,TIMESTEPS=self.num_steps)

            train_X_list.append(train_X_i)

        return train_X_list
    def _prepare_data(self, seq):##20190904查看哪里有所使用
        # split into items of input_size
        seq = [np.array(seq[i * self.input_size: (i + 1) * self.input_size])
               for i in range(len(seq) // self.input_size)]

        if self.normalized:
            seq = [seq[0] / seq[0][0] - 1.0] + [
                curr / seq[i][-1] - 1.0 for i, curr in enumerate(seq[1:])]

        # split into groups of num_steps
        X = np.array([seq[i: i + self.num_steps] for i in range(len(seq) - self.num_steps)])
        y = np.array([seq[i + self.num_steps] for i in range(len(seq) - self.num_steps)])

        train_size = int(len(X) * (1.0 - self.test_ratio))
        train_X, test_X = X[:train_size], X[train_size:]
        train_y, test_y = y[:train_size], y[train_size:]
        return train_X, train_y, test_X, test_y
    # In[]
    def _generate_one_epoch(self, batch_size):
        num_batches = int(len(self.train_X)) // batch_size #计算num_batches
        if batch_size * num_batches < len(self.train_X):#避免由于整除舍去小数后无法完全覆盖所有样本
            num_batches += 1

        batch_indices = list(range(num_batches)) 
        random.shuffle(batch_indices)#将序列的所有元素随机排序
        for j in batch_indices:
            batch_X = self.train_X[j * batch_size: (j + 1) * batch_size]
            batch_y = self.train_y[j * batch_size: (j + 1) * batch_size]
            assert set(map(len, batch_X)) == {self.num_steps}
            yield batch_X, batch_y#这样在一个epoch内，取出的batches可以刚好自动覆盖完所有的数据集

# In[]
if __name__=="__main__":
    # In[]

    #unit_DataFrame=pd.read_csv(scaled_train_path,header=None,encoding='utf-8')
    # In[]
    RUL_Data=RULDataSet()
    train_X_list=RUL_Data.train_X_list
    train_Y_list=RUL_Data.train_Y_list