# -*- coding: utf-8 -*-
'''
                        2019_03_09 Written by Bo Li
'''

#region import
import numpy as np
import pandas as pd
import pickle
import bls
from decimal import Decimal
from copy import deepcopy
from multiprocessing.pool import Pool
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
#endregion

def bls_loop(loop_time=200, data_rprice_frame = '',random_row = 30): # for eigenvalues not for portfolio
	'''
	Note: it is different form bls_loop_port, data_rprice_frame is different
	'''
	data = deepcopy(data_rprice_frame)
	label = deepcopy(data.iloc[:, -5]).values 
	data_train = deepcopy(data.iloc[:, 0:-5]).values
	traindata = data_train[0:random_row,:] #前30行
	trainlabel = data_train[0:random_row,-5] #前30行
	testdata = data_train[random_row, :].reshape(-1, traindata.shape[1])  # 注意 119列，测试数据必须是二维数据！！！
	predicted_list = []
	for i in range(0, loop_time):
		bls_model = bls.broadNet(map_num=10,
							 enhance_num=10,
							 map_function='relu',
							 enhance_function='relu',
							 batchsize=10)
		bls_model.fit(traindata, trainlabel)
		predictlabel = bls_model.predict(testdata)
		predicted_list.append(predictlabel[0])
		print(i+1)
	return np.array(predicted_list).reshape(-1,1)

def parallel_bls_loop(loop_time=200, data_list = ''):
	'''
	:param data_list: relative price frame list
	'''
	'''
		pool = ThreadPool(20)# parallel,https://www.cnblogs.com/wangxusummer/p/4835929.html
		func = partial(bls_loop, loop_time)#bls_loop中有两个参数，先传入一个参数再放入到map中
		results_list = pool.map(func, data_list)
		del pool
	'''


	func = partial(bls_loop, loop_time)  # bls_loop中有两个参数，先传入一个参数再放入到map中
	results_list = list(map(func, data_list))
	results_array = results_list[0]
	for i in range(1,len( results_list ) ):
		results_array = np.concatenate((results_array, results_list[i]), axis=1)
	return results_array

def Sample_return(data_array, sample_time=500):
	'''
	:param data_array: numpy ndarray, to sample with,
	'''
	data_frame = pd.DataFrame(data_array)
	data_frame_sample = data_frame.sample(n=sample_time,axis=0,replace=True)
	result_array = np.concatenate( (data_array, data_frame_sample.values), axis=0)
	return result_array

if __name__ == '__main__':
	#First Step
	#region Load the data dict begin

	File_LoadPath =......
	This_file = ......
	f = open(File_LoadPath + This_file, 'rb')
	data_dic = pickle.load(f)
	f.close()
	
	data_rprice_dic = deepcopy(data_dic['zd_x_dict'])
	data_rprice_list = list(data_rprice_dic.values())#将字典的values转换成列表
	predict_array = parallel_bls_loop(loop_time=50, data_list = data_rprice_list)#parallel multi assets
	predict_sample_array = Sample_return(data_array=predict_array, sample_time=50)
	eigen_val, eigen_vec = np.linalg.eig(np.corrcoef(predict_sample_array.T))
	dis_nor = np.random.normal(0, 1, predict_sample_array.T.shape[0]* predict_sample_array.T.shape[1])
	dis_nor = dis_nor.reshape(predict_sample_array.T.shape[0], predict_sample_array.T.shape[1])
	eigen_val_ran, eigen_vec_ran = np.linalg.eig(np.corrcoef(dis_nor))
	eigen_val_30min, eigen_vec_30min = eigen_val, eigen_vec
	eigen_val_60min, eigen_vec_60min = eigen_val, eigen_vec
	eigen_val_ran   # random matrix
	eigen_vec_ran   # random matrix
	bls_eigen_dics = dict()
	bls_eigen_dics['eigen_val_30min'] = eigen_val_30min
	bls_eigen_dics['eigen_vec_30min'] = eigen_vec_30min
	bls_eigen_dics['eigen_val_60min'] = eigen_val_60min
	bls_eigen_dics['eigen_vec_60min'] = eigen_vec_60min
	bls_eigen_dics['eigen_val_ran'] = eigen_val_ran
	bls_eigen_dics['eigen_vec_ran'] = eigen_vec_ran
	File_SavePath = ......
	This_file = ......
	f = open(File_SavePath+This_file, 'wb')
	pickle.dump(bls_eigen_dics, f)
	f.close()
	File_LoadPath = ......
	This_file = ......

	f = open(File_LoadPath + This_file, 'rb')
	data_dic_processed = pickle.load(f)
	f.close()
	eigen_val_ran = data_dic_processed['eigen_val_ran']
	eigen_vec_ran = data_dic_processed['eigen_vec_ran']
	eigen_val_30min = data_dic_processed['eigen_val_30min']
	eigen_val_60min = data_dic_processed['eigen_val_60min']
	l1 = len(  eigen_val_30min[np.where(eigen_val_30min > max(eigen_val_ran) )] )
	l2 = len(  eigen_val_30min[np.where(eigen_val_30min < min(eigen_val_ran) )] )
	(l1 + l2)/float(len(eigen_val_30min))

	l3 = len(  eigen_val_60min[np.where(eigen_val_60min > max(eigen_val_ran) )] )
	l4 = len(  eigen_val_60min[np.where(eigen_val_60min < min(eigen_val_ran) )] )
	(l3 + l4)/float(len(eigen_val_60min))

	sns.set_style("ticks")  
	plt.figure(figsize=(8,4),dpi=300)  
	sns.distplot(eigen_val_30min, hist=False, kde=True, rug=False,   
	kde_kws={"color":"lightcoral", "lw":1},  
	rug_kws={'color':'lightcoral','alpha':1, 'lw':2}, label='Empirical 30min')
	sns.distplot(eigen_val_60min, hist=False, kde=True, rug=False,   ）
	kde_kws={"color":"lightseagreen", "lw":1},  
	rug_kws={'color':'lightseagreen','alpha':1, 'lw':2}, label='Empirical 60min')

	sns.distplot(eigen_val_ran, hist=False, kde=True, rug=False,  
	kde_kws={"color":"LightSkyBlue", "lw":1 },  
	rug_kws={'color':'LightSkyBlue','alpha':1, 'lw':2}, label='Theoretical')
	plt.ylim(0, 1)
	plt.xlabel("Eigenvalues")
	plt.ylabel("Distribution")
	plt.title("R50I") 
	plt.show()
	plt.close()
