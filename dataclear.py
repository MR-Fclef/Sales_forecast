import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import scipy.io as sio
import gc
# read csv
data_order = pd.read_csv('Sales_Forecast_Qualification/t_order.csv')
data_ads = pd.read_csv('Sales_Forecast_Qualification/t_ads.csv')
data_comment = pd.read_csv('Sales_Forecast_Qualification/t_comment.csv')
data_sales_sum = pd.read_csv('Sales_Forecast_Qualification/t_sales_sum.csv')
data_product = pd.read_csv('Sales_Forecast_Qualification/t_product.csv')

# sort shop_id and dt & deal with data_order
data_order.sort_values(by=['shop_id','ord_dt'],ascending=[1,1],inplace=True)
data_sales_sum.sort_values(by=['shop_id','dt'],ascending=[1,1],inplace=True)
# print the sorted data
# print data_order
# print data_sales_sum

data_order.set_index("ord_dt", inplace=True)
data_order.index = pd.DatetimeIndex(data_order.index)
shop_num=3000;
sale_mont=[[0 for col in range(9)] for row in range(shop_num)];
rtn_amt=[[0 for col in range(9)] for row in range(shop_num)];
offer_amt=[[0 for col in range(9)] for row in range(shop_num)];
ord_cnt=[[0 for col in range(9)] for row in range(shop_num)];
user_cnt=[[0 for col in range(9)] for row in range(shop_num)];
rtn_cnt=[[0 for col in range(9)] for row in range(shop_num)];
offer_cnt=[[0 for col in range(9)] for row in range(shop_num)];

for i in range(0,shop_num):
    print i+1
    data_order_salesum11= data_order['2016-08-01':'2016-8-31']
    shopid_11 = data_order_salesum11[['sale_amt','rtn_amt','offer_amt','ord_cnt','user_cnt','offer_cnt','rtn_cnt']][data_order_salesum11.shop_id==i+1].apply(sum)
    data_order_salesum12= data_order['2016-09-01':'2016-09-30']
    shopid_12 = data_order_salesum12[['sale_amt','rtn_amt','offer_amt','ord_cnt','user_cnt','offer_cnt','rtn_cnt']][data_order_salesum12.shop_id==i+1].apply(sum)
    data_order_salesum13= data_order['2016-10-01':'2016-10-31']
    shopid_13 = data_order_salesum13[['sale_amt','rtn_amt','offer_amt','ord_cnt','user_cnt','offer_cnt','rtn_cnt']][data_order_salesum13.shop_id==i+1].apply(sum)
    data_order_salesum14= data_order['2016-11-01':'2016-11-30']
    shopid_14 = data_order_salesum14[['sale_amt','rtn_amt','offer_amt','ord_cnt','user_cnt','offer_cnt','rtn_cnt']][data_order_salesum14.shop_id==i+1].apply(sum)
    data_order_salesum15= data_order['2016-12-01':'2016-12-31']
    shopid_15 = data_order_salesum15[['sale_amt','rtn_amt','offer_amt','ord_cnt','user_cnt','offer_cnt','rtn_cnt']][data_order_salesum15.shop_id==i+1].apply(sum)
    data_order_salesum16= data_order['2017-01-01':'2017-01-31']
    shopid_16 = data_order_salesum16[['sale_amt','rtn_amt','offer_amt','ord_cnt','user_cnt','offer_cnt','rtn_cnt']][data_order_salesum16.shop_id==i+1].apply(sum)
    data_order_salesum17= data_order['2017-02-01':'2017-02-28']
    shopid_17 = data_order_salesum17[['sale_amt','rtn_amt','offer_amt','ord_cnt','user_cnt','offer_cnt','rtn_cnt']][data_order_salesum17.shop_id==i+1].apply(sum)
    data_order_salesum18= data_order['2017-03-01':'2017-03-31']
    shopid_18 = data_order_salesum18[['sale_amt','rtn_amt','offer_amt','ord_cnt','user_cnt','offer_cnt','rtn_cnt']][data_order_salesum18.shop_id==i+1].apply(sum)
    data_order_salesum19= data_order['2017-04-01':'2017-04-30']
    shopid_19 = data_order_salesum19[['sale_amt','rtn_amt','offer_amt','ord_cnt','user_cnt','offer_cnt','rtn_cnt']][data_order_salesum19.shop_id==i+1].apply(sum)
    ##############
    # print data_order_salesum11
    # print shopid_11,shopid_12,shopid_13
    # print data_sales_sum[['dt','sale_amt_3m']][data_sales_sum.shop_id==1]
    sale_mont[i][0]=(shopid_11['sale_amt'])
    sale_mont[i][1]=(shopid_12['sale_amt'])
    sale_mont[i][2]=(shopid_13['sale_amt'])
    sale_mont[i][3]=(shopid_14['sale_amt'])
    sale_mont[i][4]=(shopid_15['sale_amt'])
    sale_mont[i][5]=(shopid_16['sale_amt'])
    sale_mont[i][6]=(shopid_17['sale_amt'])
    sale_mont[i][7]=(shopid_18['sale_amt'])
    sale_mont[i][8]=(shopid_19['sale_amt'])

    rtn_amt[i][0]=(shopid_11['rtn_amt'])
    rtn_amt[i][1]=(shopid_12['rtn_amt'])
    rtn_amt[i][2]=(shopid_13['rtn_amt'])
    rtn_amt[i][3]=(shopid_14['rtn_amt'])
    rtn_amt[i][4]=(shopid_15['rtn_amt'])
    rtn_amt[i][5]=(shopid_16['rtn_amt'])
    rtn_amt[i][6]=(shopid_17['rtn_amt'])
    rtn_amt[i][7]=(shopid_18['rtn_amt'])
    rtn_amt[i][8]=(shopid_19['rtn_amt'])

    offer_amt[i][0]=(shopid_11['offer_amt'])
    offer_amt[i][1]=(shopid_12['offer_amt'])
    offer_amt[i][2]=(shopid_13['offer_amt'])
    offer_amt[i][3]=(shopid_14['offer_amt'])
    offer_amt[i][4]=(shopid_15['offer_amt'])
    offer_amt[i][5]=(shopid_16['offer_amt'])
    offer_amt[i][6]=(shopid_17['offer_amt'])
    offer_amt[i][7]=(shopid_18['offer_amt'])
    offer_amt[i][8]=(shopid_19['offer_amt'])

    ord_cnt[i][0]=(shopid_11['ord_cnt'])
    ord_cnt[i][1]=(shopid_12['ord_cnt'])
    ord_cnt[i][2]=(shopid_13['ord_cnt'])
    ord_cnt[i][3]=(shopid_14['ord_cnt'])
    ord_cnt[i][4]=(shopid_15['ord_cnt'])
    ord_cnt[i][5]=(shopid_16['ord_cnt'])
    ord_cnt[i][6]=(shopid_17['ord_cnt'])
    ord_cnt[i][7]=(shopid_18['ord_cnt'])
    ord_cnt[i][8]=(shopid_19['ord_cnt'])

    user_cnt[i][0]=(shopid_11['user_cnt'])
    user_cnt[i][1]=(shopid_12['user_cnt'])
    user_cnt[i][2]=(shopid_13['user_cnt'])
    user_cnt[i][3]=(shopid_14['user_cnt'])
    user_cnt[i][4]=(shopid_15['user_cnt'])
    user_cnt[i][5]=(shopid_16['user_cnt'])
    user_cnt[i][6]=(shopid_17['user_cnt'])
    user_cnt[i][7]=(shopid_18['user_cnt'])
    user_cnt[i][8]=(shopid_19['user_cnt'])

    rtn_cnt[i][0]=(shopid_11['rtn_cnt'])
    rtn_cnt[i][1]=(shopid_12['rtn_cnt'])
    rtn_cnt[i][2]=(shopid_13['rtn_cnt'])
    rtn_cnt[i][3]=(shopid_14['rtn_cnt'])
    rtn_cnt[i][4]=(shopid_15['rtn_cnt'])
    rtn_cnt[i][5]=(shopid_16['rtn_cnt'])
    rtn_cnt[i][6]=(shopid_17['rtn_cnt'])
    rtn_cnt[i][7]=(shopid_18['rtn_cnt'])
    rtn_cnt[i][8]=(shopid_19['rtn_cnt'])

    offer_cnt[i][0]=(shopid_11['offer_cnt'])
    offer_cnt[i][1]=(shopid_12['offer_cnt'])
    offer_cnt[i][2]=(shopid_13['offer_cnt'])
    offer_cnt[i][3]=(shopid_14['offer_cnt'])
    offer_cnt[i][4]=(shopid_15['offer_cnt'])
    offer_cnt[i][5]=(shopid_16['offer_cnt'])
    offer_cnt[i][6]=(shopid_17['offer_cnt'])
    offer_cnt[i][7]=(shopid_18['offer_cnt'])
    offer_cnt[i][8]=(shopid_19['offer_cnt'])
    gc.collect()
print sale_mont
sale_mont = np.array(sale_mont)
nan_where = np.isnan(sale_mont)
sale_mont[nan_where] = 0
sale_mont = pd.DataFrame(np.array(sale_mont,dtype=int))
sale_mont.to_csv('sale_amt.csv')

rtn_amt = np.array(rtn_amt)
nan_where = np.isnan(rtn_amt)
rtn_amt[nan_where] = 0
rtn_amt = pd.DataFrame(np.array(rtn_amt,dtype=int))
rtn_amt.to_csv('rtn_amt.csv')

offer_amt = np.array(offer_amt)
nan_where = np.isnan(offer_amt)
offer_amt[nan_where] = 0
offer_amt = pd.DataFrame(np.array(offer_amt,dtype=int))
offer_amt.to_csv('offer_amt.csv')

ord_cnt = np.array(ord_cnt)
nan_where = np.isnan(ord_cnt)
ord_cnt[nan_where] = 0
ord_cnt = pd.DataFrame(np.array(ord_cnt,dtype=int))
ord_cnt.to_csv('ord_cnt.csv')

user_cnt = np.array(user_cnt)
nan_where = np.isnan(user_cnt)
user_cnt[nan_where] = 0
user_cnt = pd.DataFrame(np.array(user_cnt,dtype=int))
user_cnt.to_csv('user_cnt.csv')

rtn_cnt = np.array(rtn_cnt)
nan_where = np.isnan(rtn_cnt)
rtn_cnt[nan_where] = 0
rtn_cnt = pd.DataFrame(np.array(rtn_cnt,dtype=int))
rtn_cnt.to_csv('rtn_cnt.csv')

offer_cnt = np.array(offer_cnt)
nan_where = np.isnan(offer_cnt)
offer_cnt[nan_where] = 0
offer_cnt = pd.DataFrame(np.array(offer_cnt,dtype=int))
offer_cnt.to_csv('offer_cnt.csv')
# numpy.savetxt("t_order_new.txt",sale_mont)
# plt.figure()
# for i in range(0,shop_num):
#     plt.plot(sale_mont[i])
# plt.show()




# deal with ads
# data_ads.sort_values(by=['shop_id','create_dt'],ascending=[1,1],inplace=True)
# data_ads.set_index("create_dt", inplace=True)
# data_ads.index = pd.DatetimeIndex(data_ads.index)
# print data_ads

