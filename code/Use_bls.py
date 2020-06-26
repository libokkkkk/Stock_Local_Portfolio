'''
                 2019_02_02 Written by Bo Li

'''

#region import
import numpy as np
import pandas as pd
import pickle
import bls
from decimal import Decimal
from copy import deepcopy
#endregion




def comp_x_zd(data_dic):# return data_for_bls
    '''
    data_dic: dataframes dict

    '''
    data_for_bls = deepcopy(data_dic)
   
    for key in data_for_bls.keys():
        temp_x = np.array([1.0001] * data_for_bls[key].shape[0])# for  x(t)/x(t-1)
        for i in range(1, data_for_bls[key].shape[0]):
            temp_x[i] = Decimal(data_for_bls[key]['close'][i]/data_for_bls[key]['pre_close'][i]).\
                quantize(Decimal('0.0000'))
        temp_x[0]=Decimal(1.0000).quantize(Decimal('0.0000'))#temp_x[0]has no pre_close value
        temp_x = pd.DataFrame(temp_x, columns = ['x_zd'])
        data_for_bls[key] = pd.concat([data_for_bls[key],temp_x], axis = 1)
    return data_for_bls


def get_x_zd_labels(data_for_bls, length_series):#return zd_x_dict
    '''
    data_for_bls: the frame 

    '''
   
    zd_x_dict = deepcopy(data_for_bls)#zd_x frame_dict
    for key in zd_x_dict.keys():
        this_key_Frame = pd.DataFrame()
        for i in range(length_series, zd_x_dict[key].shape[0]): # eg:length_series = 100
            temp_x = pd.DataFrame(zd_x_dict[key]['x_zd'][i-length_series:i].reshape(-1,1))
            this_key_Frame = pd.concat([this_key_Frame, temp_x], axis=1)
            print(i)
        this_key_Frame = this_key_Frame.T

        zd_x_dict[key] = this_key_Frame
        print(key)
  
    for key in zd_x_dict.keys():
        zd_x_dict[key]['softinc'], zd_x_dict[key]['softdec'] = 0, 0
        zd_x_dict[key]['portinc'], zd_x_dict[key]['portdec'] = 0, 0
        num_columns = zd_x_dict[key].shape[1]#num of columns : shape[1]
        for i in range(0, zd_x_dict[key].shape[0]):
            if zd_x_dict[key].iloc[i, -5] > 1:
                zd_x_dict[key].iloc[i, num_columns-4 ] = 1
                zd_x_dict[key].iloc[i, num_columns-2] = zd_x_dict[key].iloc[i, -5]

            elif zd_x_dict[key].iloc[i, -5] < 1:
                zd_x_dict[key].iloc[i, num_columns-3] = 1
                zd_x_dict[key].iloc[i, num_columns-1] = zd_x_dict[key].iloc[i, -5]

            else: #no change
                zd_x_dict[key].iloc[i, num_columns-4], zd_x_dict[key].iloc[i, num_columns-3] = 0.5, 0.5
                zd_x_dict[key].iloc[i, num_columns-2], zd_x_dict[key].iloc[i, num_columns-1] = 1, 1

    return zd_x_dict





###############################################################

def get_price_dic(data_for_bls, length_series):#return price_dict
    '''
    param data_for_bls:  see function get_x_zd_labels
    param length_series: see function get_x_zd_labels
    return: price_dict , has the price frames for each stock
    '''
    price_dict = deepcopy(data_for_bls)#zd_x frame_dict
    for key in price_dict.keys():
        this_key_Frame = pd.DataFrame()
        for i in range(length_series, price_dict[key].shape[0]):
            temp_x = pd.DataFrame(price_dict[key]['close'][i-length_series:i].reshape(-1,1))# must use reshape
            this_key_Frame = pd.concat([this_key_Frame, temp_x], axis=1)
            print(i)
        price_dict[key] = this_key_Frame.T
        print(key)
    return price_dict




def get_relative_price_frame(data_for_bls): #return relative price_frame
    '''
        param data_for_bls:  see function get_x_zd_labels
    '''
    price_dict = deepcopy(data_for_bls)#zd_x frame_dict
    this_rprice_frame = pd.DataFrame()
    for key in price_dict.keys():
        pd2 = price_dict[key].iloc[:,-1].to_frame()
        this_rprice_frame = pd.concat([this_rprice_frame, pd2], axis=1, ignore_index=True)
        print(key)
    return this_rprice_frame


if __name__ == '__main__':
    #region Data Preprocessing Begin
        # Load the data Begin
    File_LoadPath =......
    This_file = ......

    f = open(File_LoadPath + This_file, 'rb')
    data_dic = pickle.load(f)
    f.close()
    data_dic.keys()

    length_series = 120
   
    data_for_bls = comp_x_zd(data_dic)

    zd_x_dict = get_x_zd_labels(data_for_bls, length_series)
    price_dict = get_price_dic(data_for_bls, length_series)
    rprice_frame = get_relative_price_frame(data_for_bls)
    

    bls_data_dics = dict()
    bls_data_dics['zd_x_dict'] = zd_x_dict
    bls_data_dics['price_dict'] = price_dict
    bls_data_dics['rprice_frame'] = rprice_frame

    File_SavePath = ......
    This_file = ......

    f = open(File_SavePath+This_file, 'wb')
    pickle.dump(bls_data_dics, f)
    f.close()

    File_LoadPath = ......
    This_file = ......
    f = open(File_LoadPath + This_file, 'rb')
    data_dic_processed = pickle.load(f)
    f.close()

    data_dic_processed['rprice_frame'].to_csv("R50I_60.csv", index=False, header=0)
        # load end

