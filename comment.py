import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import math
import scipy.io as sio
import gc

data_comment = pd.read_csv('Sales_Forecast_Qualification/t_comment.csv')
data_comment.sort_values(by=['shop_id','create_dt'],ascending=[1,1],inplace=True)
print data_comment

data_comment.set_index("create_dt", inplace=True)
data_comment.index = pd.DatetimeIndex(data_comment.index)
shop_num=3000;
bad_comment=[[0 for col in range(9)] for row in range(shop_num)];
dis_comment=[[0 for col in range(9)] for row in range(shop_num)];
good_comment=[[0 for col in range(9)] for row in range(shop_num)];

for i in range(0,shop_num):
    print i+1
    data_order_salesum11= data_comment['2016-08-01':'2016-8-31']
    shopid_11 = data_order_salesum11[['bad_num','dis_num','good_num']][data_order_salesum11.shop_id==i+1].apply(sum)
    data_order_salesum12= data_comment['2016-09-01':'2016-09-30']
    shopid_12 = data_order_salesum12[['bad_num','dis_num','good_num']][data_order_salesum12.shop_id==i+1].apply(sum)
    data_order_salesum13= data_comment['2016-10-01':'2016-10-31']
    shopid_13 = data_order_salesum13[['bad_num','dis_num','good_num']][data_order_salesum13.shop_id==i+1].apply(sum)
    data_order_salesum14= data_comment['2016-11-01':'2016-11-30']
    shopid_14 = data_order_salesum14[['bad_num','dis_num','good_num']][data_order_salesum14.shop_id==i+1].apply(sum)
    data_order_salesum15= data_comment['2016-12-01':'2016-12-31']
    shopid_15 = data_order_salesum15[['bad_num','dis_num','good_num']][data_order_salesum15.shop_id==i+1].apply(sum)
    data_order_salesum16= data_comment['2017-01-01':'2017-01-31']
    shopid_16 = data_order_salesum16[['bad_num','dis_num','good_num']][data_order_salesum16.shop_id==i+1].apply(sum)
    data_order_salesum17= data_comment['2017-02-01':'2017-02-28']
    shopid_17 = data_order_salesum17[['bad_num','dis_num','good_num']][data_order_salesum17.shop_id==i+1].apply(sum)
    data_order_salesum18= data_comment['2017-03-01':'2017-03-31']
    shopid_18 = data_order_salesum18[['bad_num','dis_num','good_num']][data_order_salesum18.shop_id==i+1].apply(sum)
    data_order_salesum19= data_comment['2017-04-01':'2017-04-30']
    shopid_19 = data_order_salesum19[['bad_num','dis_num','good_num']][data_order_salesum19.shop_id==i+1].apply(sum)
    ##############
    # print data_order_salesum11
    # print shopid_11,shopid_12,shopid_13
    # print data_sales_sum[['dt','sale_amt_3m']][data_sales_sum.shop_id==1]
    bad_comment[i][0]=(shopid_11['bad_num'])
    bad_comment[i][1]=(shopid_12['bad_num'])
    bad_comment[i][2]=(shopid_13['bad_num'])
    bad_comment[i][3]=(shopid_14['bad_num'])
    bad_comment[i][4]=(shopid_15['bad_num'])
    bad_comment[i][5]=(shopid_16['bad_num'])
    bad_comment[i][6]=(shopid_17['bad_num'])
    bad_comment[i][7]=(shopid_18['bad_num'])
    bad_comment[i][8]=(shopid_19['bad_num'])
    dis_comment[i][0] = (shopid_11['dis_num'])
    dis_comment[i][1] = (shopid_12['dis_num'])
    dis_comment[i][2] = (shopid_13['dis_num'])
    dis_comment[i][3] = (shopid_14['dis_num'])
    dis_comment[i][4] = (shopid_15['dis_num'])
    dis_comment[i][5] = (shopid_16['dis_num'])
    dis_comment[i][6] = (shopid_17['dis_num'])
    dis_comment[i][7] = (shopid_18['dis_num'])
    dis_comment[i][8] = (shopid_19['dis_num'])
    good_comment[i][0] = (shopid_11['good_num'])
    good_comment[i][1] = (shopid_12['good_num'])
    good_comment[i][2] = (shopid_13['good_num'])
    good_comment[i][3] = (shopid_14['good_num'])
    good_comment[i][4] = (shopid_15['good_num'])
    good_comment[i][5] = (shopid_16['good_num'])
    good_comment[i][6] = (shopid_17['good_num'])
    good_comment[i][7] = (shopid_18['good_num'])
    good_comment[i][8] = (shopid_19['good_num'])
    gc.collect()
print bad_comment
numpy.savetxt("bad_comment.txt",bad_comment)
numpy.savetxt("dis_comment.txt",dis_comment)
numpy.savetxt("good_comment.txt",good_comment)
