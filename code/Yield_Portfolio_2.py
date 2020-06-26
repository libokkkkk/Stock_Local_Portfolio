# -*- coding: utf-8 -*-
'''
                        2019_03_26 Written by Bo Li
'''

from Eigen_analysis import * #需要用到该代码文件中定义的函数
import math
import numpy as np
import bls
import pickle
from copy import deepcopy
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
from sklearn.decomposition import PCA
import scipy.optimize as sco

def get_portfolio(loop_time=200, rprice_frame = '', window_size = 240, period_index = 241):
    '''
    :param rprice_frame: relative price frame
    :param loop_time: number of bls yielding the relative price
    :window_size: the size of window to match model
    :period_index: the period index to perform the algorithm
 
    '''
    temp_frame = rprice_frame.T # temp_frame: num of assets * periods
    data_rprice_row_list = []
    for i in range(0, temp_frame.shape[0]):
        data_rprice_row_list.append(temp_frame.iloc[i, :])
    multi_avg_price = np.zeros(shape = (temp_frame.shape[0],1))
    multi_port = np.zeros(shape = (temp_frame.shape[0],1))
    for p_index in range(period_index, temp_frame.shape[1]+1): 

        all_relative_price = parallel_bls_loop_port(loop_time = loop_time,\
                                                    data_rprice_row_list = data_rprice_row_list,\
                                                    window_size = window_size,
                                                    period_index = p_index)
        # all_relative_price: looptimes * num of assets; type: ndarray
        max_price_present, recon_covari_matrix_present =\
                    eigen_analysis(all_relative_price=all_relative_price, sample_times=loop_time)
        delta = 10.0 
        '''
        temp_matrix = np.matrix(recon_covari_matrix_present)
        if np.linalg.det(recon_covari_matrix_present)!=0:
            port_value = np.dot(np.linalg.inv(delta * recon_covari_matrix_present),max_price_present)
        else:
            port_value = np.dot(np.linalg.pinv(delta * recon_covari_matrix_present), max_price_present)

        '''
        print('开始优化')
        n_assets = max_price_present.shape[0]

        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bnds = (tuple((0, 1) for x in range(n_assets)))
        func_1 = partial(opt_port_aim, max_price_present, recon_covari_matrix_present)
        opt_port = sco.minimize(func_1, n_assets * [1. / n_assets, ],
                            method='SLSQP',
                            bounds=bnds, constraints=cons)
        port_value = opt_port.x.reshape(n_assets, -1)
        if p_index == period_index:
            multi_avg_price = deepcopy(max_price_present)
            multi_port = deepcopy(port_value)
        else:
            multi_avg_price = np.concatenate((multi_avg_price, max_price_present), axis=1)
            multi_port = np.concatenate((multi_port, port_value), axis=1)

    return multi_port



def bls_loop_port(loop_time = 200, window_size = 240, period_index = 241,data_rprice_row = ''): 
    '''
    :param data_rprice_row:  a row in relative price frame, each row series represents an asset
    :param loop_time:  number of bls yielding the relative price
    :return: predict relative prices
    '''
    data = deepcopy(data_rprice_row.values.reshape(1, -1) )
    train_sample_num = 10 
    traindata = np.array( [0.0] *train_sample_num * (window_size-1)  ).reshape(train_sample_num ,-1)
    trainlabel = np.array([0.0]*train_sample_num).reshape(train_sample_num,-1)# train_sample_num * 1
    testdata = deepcopy(data[0, period_index-window_size : period_index-1].reshape(1,-1) )
    cor_array = np.array([0.0]*train_sample_num).reshape(train_sample_num,-1)
   
    for j in range(0, period_index-window_size-1): 
        if j <= 9:
            traindata[j,:] = deepcopy(data[0,j:j+window_size-1])
            trainlabel[j,0] = data[0,j+window_size-1]   
            cor_array[j,0] = np.corrcoef(testdata, traindata[j,:])[0,1]
        else:
            temp_cor = np.corrcoef(testdata, data[0,j:j+window_size-1])[0, 1]
            if temp_cor > min(cor_array):
                trainlabel[np.argmin(cor_array), 0] = data[0, j+window_size-1]
                traindata[np.argmin(cor_array), :] = deepcopy(data[0,j:j+window_size-1])
                cor_array[np.argmin(cor_array), 0] = temp_cor
    predicted_list = []
    for i in range(0, loop_time):
        bls_model = bls.broadNet(map_num=10,
							 enhance_num=10,
							 map_function='relu',
							 enhance_function='relu',
							 batchsize=10)# can be alter the parameters
        bls_model.fit(traindata, trainlabel)
        predictlabel = bls_model.predict(testdata)
        predicted_list.append(predictlabel[0,0]) # 2 维,应该predicted_list.append(predictlabel[0,0])

    return np.array(predicted_list).reshape(-1,1)#


def parallel_bls_loop_port(loop_time=200, data_rprice_row_list = '', window_size = 240, period_index = 241):
    '''
    :param data_rprice_row_list: relative price row list, each element is a relative price row
    :param loop_time: number of bls yielding the relative price
    :return: predict relative prices array 
    '''
  
    '''##parallel version
    pool = ThreadPool(20)
    func = partial(bls_loop_port, loop_time, window_size, period_index)
    results_list = pool.map(func, data_rprice_row_list)
    del pool
    '''

    ## unparallel version
    print('period_index is:', period_index)
    func = partial(bls_loop_port, loop_time, window_size, period_index) 
    results_list = list(map(func, data_rprice_row_list))

    results_array = results_list[0]
    for i in range(1, len(results_list) ):
        results_array = np.concatenate((results_array, results_list[i]), axis=1)
    return results_array

def eigen_analysis(all_relative_price = '', sample_times = 200):
    '''
   
    :param all_relative_price: looptimes * num of assets; type: ndarray
    :param sample_times: samples times on all_relative_price
    '''

    #The  definition of this function Sample_return is in Eigen_analysis.py

    predict_sample_array = Sample_return(data_array=all_relative_price, sample_time=sample_times)  
    tiny = 1e-15
    for i in range(0, predict_sample_array.T.shape[0]):
        predict_sample_array.T[i, 1] = predict_sample_array.T[i, 1] + tiny
    eigen_val, eigen_vec = np.linalg.eig(np.corrcoef(predict_sample_array.T))  # np.cov(x)协方差
    eigen_val, eigen_vec = eigen_val.real, eigen_vec.real
    dis_nor = np.random.normal(0, 1, predict_sample_array.T.shape[0]* predict_sample_array.T.shape[1])
    dis_nor = dis_nor.reshape(predict_sample_array.T.shape[0], predict_sample_array.T.shape[1])
    tiny = 1e-15
    for i in range(0,dis_nor.shape[0]):
        dis_nor[i,1] = dis_nor[i,1]+tiny
    eigen_val_ran, eigen_vec_ran = np.linalg.eig(np.corrcoef(dis_nor))#for 随机矩阵
    eigen_val_ran, eigen_vec_ran = eigen_val_ran.real, eigen_vec_ran.real 
    eigen_val_filterindex = np.where((eigen_val < max(eigen_val_ran) )\
                                & (eigen_val > min(eigen_val_ran)))
    eigen_val_filterindex = eigen_val_filterindex[0]
    pca_F1 = PCA(n_components=1)

    reduced_x = pca_F1.fit_transform(eigen_vec[:, eigen_val_filterindex].T) 
    F1_imp = pca_F1.components_ 
    imp_square = np.array(list(map(lambda x: x**2 , list(F1_imp[0]))) )# 
    Corrected_diag_imp = np.diag(1-imp_square/sum(imp_square))
    Corrected_eigen_vec = np.dot(Corrected_diag_imp, eigen_vec[:, eigen_val_filterindex]).real
    k =0
    for i in eigen_val_filterindex:
        eigen_vec[:,i] = Corrected_eigen_vec[:,k]
        k = k+1
    for i in eigen_val_filterindex:
        eigen_val[i] = min(eigen_val_ran)
    recon_cor_matrix = np.dot(np.dot(eigen_vec,np.diag(eigen_val) ), np.linalg.inv(eigen_vec)).real
    for i in range(0, recon_cor_matrix.shape[0]):
        recon_cor_matrix[i,i]=1
    recon_cor_matrix[recon_cor_matrix < -1] = -1
    recon_cor_matrix[recon_cor_matrix > 1] = 1
    temp_std = np.std(predict_sample_array, axis=0,ddof=1) # shape:1 * num of assets
    for i in range(0, recon_cor_matrix.shape[0]):
        for j in range(0, recon_cor_matrix.shape[1]):
            recon_cor_matrix[i,j] = recon_cor_matrix[i,j] * temp_std[i] * temp_std[j]

    recon_covari_matrix = recon_cor_matrix
    max_price = np.mean(predict_sample_array.T, axis=1).reshape(-1, 1)
    return max_price, recon_covari_matrix 
def portfolio_aim(max_price_present, recon_covari_matrix_present, weights):
    '''
    :param weights: 
    :return: 
    '''
    weights = np.array(weights)
    port_returns = np.dot(weights.T, max_price_present.real)#马尔科维兹公式前一半

    port_variance = np.dot( np.dot(weights.T, recon_covari_matrix_present.real), weights)  
    return np.array([port_returns, port_variance, port_returns * 1.0 - 0.24*np.dot(weights.T,weights)-0.15*port_variance] )

def opt_port_aim(max_price_present, recon_covari_matrix_present, weights):
    return -portfolio_aim(max_price_present, recon_covari_matrix_present,weights)[2]

if __name__ == '__main__':
    File_LoadPath = ......
    This_file = ......
    f = open(File_LoadPath + This_file, 'rb')
    data_dic = pickle.load(f)
    f.close()

    rprice_frame = data_dic['rprice_frame']
    period_begin = int(rprice_frame.shape[0]/2)
    port_frame = get_portfolio(loop_time=200, rprice_frame=rprice_frame, window_size=120, period_index=period_begin+1)
    rprice_port_frame = dict()
    rprice_port_frame['port_frame'] = port_frame
    File_SavePath = .......
    This_file = ......
    f = open(File_SavePath + This_file, 'wb')
    pickle.dump(rprice_port_frame, f)
    f.close()
    # 序列化 结束保存
