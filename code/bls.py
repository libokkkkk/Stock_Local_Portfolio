'''
edited  by bo li
'''

#region import
import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime
from copy import deepcopy
import pickle
#endregion

def show_accuracy(predictLabel,Label):
    Label = np.ravel(Label).tolist()
    predictLabel = predictLabel.tolist()
    count = 0
    
    for i in range(len(Label)):
        if Label[i] == predictLabel[i]:
            count += 1
    return (round(count/len(Label),5))

class node_generator(object):
	def __init__(self, isenhance = False):
		self.Wlist = []
		self.blist = []
		self.function_num = 0
		self.isenhance = isenhance

	def sigmoid(self, x):
		return 1.0/(1 + np.exp(-x))
	def relu(self, x):
		return np.maximum(x, 0)
	def tanh(self, x):
		return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
	def linear(self, x):
		return x
	def orth(self, W):
		"""
		
		"""
		for i in range(0, W.shape[1]):
			w = np.mat(W[:,i].copy()).T
			w_sum = 0
			for j in range(i):
				wj = np.mat(W[:,j].copy()).T
				w_sum += (w.T.dot(wj))[0,0]*wj
			w -= w_sum
			w = w/np.sqrt(w.T.dot(w))
			W[:,i] = np.ravel(w)

		return W

	def generator(self, shape, times):
		for i in range(times):
			W = 2*np.random.random(size=shape)-1
			if self.isenhance == True:
				W = self.orth(W)   # 
			b = 2*np.random.random() -1
			yield (W, b)

	def generator_nodes(self, data, times, batchsize, function_num):
		
		self.Wlist = [elem[0] for elem in self.generator((data.shape[1], batchsize), times)]
		self.blist = [elem[1] for elem in self.generator((data.shape[1], batchsize), times)]
		
		self.function_num = {'linear':self.linear,
						'sigmoid': self.sigmoid,
						'tanh':self.tanh,
						'relu':self.relu }[function_num]  
		
		nodes = self.function_num(data.dot(self.Wlist[0]) + self.blist[0])
		for i in range(1, len(self.Wlist)):
			nodes = np.column_stack((nodes, self.function_num(data.dot(self.Wlist[i])+self.blist[i])))
		#print("nodes的维度",nodes.shape)
		return nodes

	def transform(self,testdata):
		testnodes = self.function_num(testdata.dot(self.Wlist[0])+self.blist[0])
		for i in range(1,len(self.Wlist)):
			testnodes = np.column_stack((testnodes, self.function_num(testdata.dot(self.Wlist[i])+self.blist[i])))
		return testnodes

class scaler:
    def __init__(self):
        self._mean = 0
        self._std = 0
    
    def fit_transform(self,traindata):
        self._mean = traindata.mean(axis = 0)
        self._std = traindata.std(axis = 0)
        return (traindata-self._mean)/(self._std+0.001)
    
    def transform(self,testdata):
        return (testdata-self._mean)/(self._std+0.001)

class broadNet(object):
	def __init__(self, map_num=10,enhance_num=10,map_function='linear',enhance_function='linear',batchsize='auto'):
		self.map_num = map_num
		self.enhance_num = enhance_num
		self.batchsize = batchsize
		self.map_function = map_function
		self.enhance_function = enhance_function

		self.W = 0
		self.pseudoinverse = 0
		self.normalscaler = scaler()
		
		self.onehotencoder = preprocessing.OneHotEncoder(sparse = False)
		self.mapping_generator = node_generator()
		self.enhance_generator = node_generator(isenhance = True)
		self.label_1 = []
		self.label_2 = []

	def fit(self, data, label):
		if self.batchsize == 'auto':
			self.batchsize = data.shape[1]
		self.label_1 = label
		#data = self.normalscaler.fit_transform(data)
		
		self.label_2 = label
		mappingdata = self.mapping_generator.generator_nodes(data, self.map_num, self.batchsize,self.map_function)
		enhancedata = self.enhance_generator.generator_nodes(mappingdata, self.enhance_num, self.batchsize,self.enhance_function)

		inputdata = np.column_stack((mappingdata, enhancedata))
		pseudoinverse = np.linalg.pinv(inputdata)
		self.W = pseudoinverse.dot(label)


	def decode(self,Y_onehot):
		Y = []
		for i in range(Y_onehot.shape[0]):
			lis = np.ravel(Y_onehot[i,:]).tolist()
			Y.append(lis.index(max(lis)))
		return np.array(Y)

	def accuracy(self,predictlabel,label):
		label = np.ravel(label).tolist()
		predictlabel = predictlabel.tolist()
		count = 0
		for i in range(len(label)):
			if label[i] == predictlabel[i]:
				count += 1
		return (round(count/len(label),5))

	def predict(self, testdata):
		
		test_mappingdata = self.mapping_generator.transform(testdata)
		test_enhancedata = self.enhance_generator.transform(test_mappingdata)

		test_inputdata = np.column_stack((test_mappingdata,test_enhancedata))    
		#return self.decode(test_inputdata.dot(self.W)) 
		return test_inputdata.dot(self.W)

if __name__ == '__main__':
	#test
	File_LoadPath = ......  #need to set 
	This_file = ......   #need to set 
	f = open(File_LoadPath + This_file, 'rb')
	data_dic = pickle.load(f)
	f.close()
	data_dic.keys()
		
	data = .........
	label = deepcopy(data.iloc[:, data.shape[1] - 4:]).values
	data = deepcopy(data.iloc[:, 0:data.shape[1] - 4]).values

	traindata,testdata,trainlabel,testlabel = train_test_split(data, label,test_size=0.2,random_state = 0)

	bls = broadNet(map_num = 10,\
               		enhance_num = 10,\
               		map_function = 'relu',\
               		enhance_function = 'relu',\
					batchsize = 100)
	starttime = datetime.datetime.now()
	bls.fit(traindata, trainlabel)
	endtime = datetime.datetime.now()
	predictlabel = bls.predict(testdata)#should be shape: (,4)

