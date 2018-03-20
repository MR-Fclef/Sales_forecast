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
from sklearn.decomposition import PCA
import xgboost as xgb
import scipy.io as sio

sale_amt = pd.read_csv('data_k7.csv')

# clear the data fo sale_sum
shopid_k6 = sale_amt[0:]['9']
print shopid_k6
data_sales_sum = pd.read_csv('Sales_Forecast_Qualification/t_sales_sum.csv')
data_sales_sum.sort_values(by=['shop_id','dt'],ascending=[1,1],inplace=True)
data_sales_sum.set_index("dt", inplace=True)
data_sales_sum.index = pd.DatetimeIndex(data_sales_sum.index)
#for i in range(1,3001):
df1 = data_sales_sum[data_sales_sum['shop_id']==2205].drop_duplicates('sale_amt_3m')
df2 = data_sales_sum[data_sales_sum['shop_id']==2129].drop_duplicates('sale_amt_3m')
data_sales_sum = data_sales_sum[(True-data_sales_sum['shop_id'].isin([2205,2129]))]
data_sales_sum = pd.concat([data_sales_sum,df1,df2])
data_trainY = data_sales_sum
# print data_trainY
trainy = pd.DataFrame(columns=[0])
for i in range(0,len(sale_amt)):
    df1 = data_trainY[data_trainY['shop_id']==shopid_k6[i]+1][2:7]['sale_amt_3m']
    trainy = pd.concat([trainy,df1])
# print trainy

testy = pd.DataFrame(columns=[0])
for i in range(0,len(sale_amt)):
    df1 = data_trainY[data_trainY['shop_id']==shopid_k6[i]+1][7:8]['sale_amt_3m']
    testy = pd.concat([testy,df1])
# print testy

# print data_sales_sum
# select the data of ki
ads_consume = np.loadtxt("ads_consume.txt")
bad_comment = np.loadtxt("bad_comment.txt")
dis_comment = np.loadtxt("dis_comment.txt")
good_comment = np.loadtxt("good_comment.txt")

offer_amt = pd.read_csv('offer_amt.csv')
offer_cnt = pd.read_csv('offer_cnt.csv')
ord_cnt = pd.read_csv('ord_cnt.csv')
rtn_cnt = pd.read_csv('rtn_cnt.csv')
rtn_amt = pd.read_csv('rtn_amt.csv')
user_cnt = pd.read_csv('user_cnt.csv')

offer_amt = offer_amt.iloc[:,1:]
offer_cnt = offer_cnt.iloc[:,1:]
ord_cnt = ord_cnt.iloc[:,1:]
rtn_cnt = rtn_cnt.iloc[:,1:]
rtn_amt = rtn_amt.iloc[:,1:]
user_cnt = user_cnt.iloc[:,1:]

offer_amt_k = np.array(offer_amt)[shopid_k6]
offer_cnt_k = np.array(offer_cnt)[shopid_k6]
ord_cnt_k = np.array(ord_cnt)[shopid_k6]
rtn_cnt_k = np.array(rtn_cnt)[shopid_k6]
rtn_amt_k = np.array(rtn_amt)[shopid_k6]
user_cnt_k = np.array(user_cnt)[shopid_k6]

# print offer_amt
# print np.array(offer_amt)[shopid_k6]

ads_consume = np.nan_to_num(ads_consume)
ads_consume_k6 = ads_consume[shopid_k6]
bad_comment_k6 = bad_comment[shopid_k6]
dis_comment_k6 = dis_comment[shopid_k6]
good_comment_k6 = good_comment[shopid_k6]
# concat the data
#sale_amt ads bad_comm dis_comm good_comm offer_amt offer_cnt ord_cnt rtn_cnt rtn_amt user_cnt
data_trainX = pd.DataFrame(columns=[0,1,2,3,4,5,6,7,8,9,10])
for i in range(0,len(sale_amt)):
    df1 = pd.DataFrame(sale_amt.iloc[i:i+1,1:10])
    df2 = pd.DataFrame(ads_consume_k6[i:i+1][0:9])
    # print df1
    df4 = df2.reindex(columns=['0','1','2','3','4','5','6','7','8'])
    df4.iloc[0,0:] = ads_consume_k6[i:i+1][0:9]
    df1 = pd.concat([df1,df4]
                 ,ignore_index = 'true')
    # print df1
    df4.iloc[0,0:] = bad_comment_k6[i:i+1][0:9]
    df1 = pd.concat([df1,df4]
                 ,ignore_index = 'true')
    # print df1
    df4.iloc[0,0:] = dis_comment_k6[i:i+1][0:9]
    df1 = pd.concat([df1,df4]
                 ,ignore_index = 'true')
    # print df1
    df4.iloc[0,0:] = good_comment_k6[i:i+1][0:9]
    df1 = pd.concat([df1,df4]
                 ,ignore_index = 'true')

    df4.iloc[0, 0:] = offer_amt_k[i:i + 1][0:9]
    df1 = pd.concat([df1, df4]
                    , ignore_index='true')

    df4.iloc[0, 0:] = offer_cnt_k[i:i + 1][0:9]
    df1 = pd.concat([df1, df4]
                    , ignore_index='true')

    df4.iloc[0, 0:] = ord_cnt_k[i:i + 1][0:9]
    df1 = pd.concat([df1, df4]
                    , ignore_index='true')

    df4.iloc[0, 0:] = rtn_cnt_k[i:i + 1][0:9]
    df1 = pd.concat([df1, df4]
                    , ignore_index='true')

    df4.iloc[0, 0:] = rtn_amt_k[i:i + 1][0:9]
    df1 = pd.concat([df1, df4]
                    , ignore_index='true')

    df4.iloc[0, 0:] = user_cnt_k[i:i + 1][0:9]
    df1 = pd.concat([df1, df4]
                    , ignore_index='true')


    # data_train = df1.T
    # print df1.T.columns
    data_trainX = pd.concat([data_trainX,df1.T]
                 ,ignore_index = 'true')
# print data_trainX

# get the tranx from the data_tranX
trainX = pd.DataFrame(columns=[0,1,2,3,4,5,6,7,8,9,10])
for i in range(0,len(sale_amt)):
    df1 = pd.DataFrame(data_trainX.iloc[9*i:9*i+5,0:])
    trainX = pd.concat([trainX,df1]
                    ,ignore_index = 'true')
print trainX

testX = pd.DataFrame(columns=[0,1,2,3,4,5,6,7,8,9,10])
for i in range(0,len(sale_amt)):
    df1 = pd.DataFrame(data_trainX.iloc[9*i+5:9*i+6,0:])
    testX = pd.concat([testX,df1]
                    ,ignore_index = 'true')
print testX

#PCA
trainX = np.nan_to_num(trainX)
trainX = np.nan_to_num(trainX)

trainX = np.array(trainX)
testX = np.array(testX)
pca = PCA(n_components=9)
new_trainX = pca.fit_transform(trainX)
new_testX = pca.fit_transform(testX)
print new_trainX


#guiyihua
x = pd.concat([pd.DataFrame(new_trainX),pd.DataFrame(new_testX)])
x = x.fillna(0)
scaler_x = MinMaxScaler(feature_range=(0, 1))
x = scaler_x.fit_transform(x.values)
trainx = x[0:len(sale_amt)*5,0:]
testx = x[len(sale_amt)*5:,0:]
print len(trainx)
print len(testx)

y = pd.concat([trainy,testy])
y = y.fillna(0)
scaler_y = MinMaxScaler(feature_range=(0, 1))
print y.values
y = scaler_y.fit_transform(y.values)
trainy = y[0:len(sale_amt)*5,0:]
testy = y[len(sale_amt)*5:,0:]
print len(trainx)
print len(trainy)
print len(testx)
print len(testy)

# XGB
dtrain = xgb.DMatrix(trainx,trainy)
print dtrain
# num_trees = 450
params = {
    "objective": "reg:linear",
    "eta": 0.15,
    # "max_depth": 8,
    # "subsample": 0.7,
    # "colsample_bytree": 0.7,
}

gbm = xgb.train(params,dtrain)

testPredict = gbm.predict(xgb.DMatrix(testx))
print np.array(testPredict).reshape(-1,1)
testPredict = scaler_y.inverse_transform(np.array(testPredict).reshape(-1,1))
testy = scaler_y.inverse_transform(testy)

# print testPredict[0:,0]
# print testy
plt.plot(testPredict[0:,0],'b')
plt.plot(testy[0:,0],'r')
plt.show()
testScore = math.sqrt(mean_squared_error(testy[0:,0], testPredict[0:,0]))
print('Test Score: %.2f RMSE' % (testScore))
