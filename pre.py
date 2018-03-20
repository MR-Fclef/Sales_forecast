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
# %matplotlib inline

# load the datatxt
data_k1 = pd.read_csv('data_k28.csv')
for i in range(0,len(data_k1)):
    plt.plot(np.array(data_k1.iloc[i:i+1,1:10]).tolist()[0])
plt.show()



# clear the data fo sale_sum
shopid_k6 = data_k1[0:]['9']
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
print data_trainY
trainy = pd.DataFrame(columns=[0])
for i in range(0,len(data_k1)):
    df1 = data_trainY[data_trainY['shop_id']==shopid_k6[i]+1][2:7]['sale_amt_3m']
    trainy = pd.concat([trainy,df1])
# print trainy

testy = pd.DataFrame(columns=[0])
for i in range(0,len(data_k1)):
    df1 = data_trainY[data_trainY['shop_id']==shopid_k6[i]+1][7:8]['sale_amt_3m']
    testy = pd.concat([testy,df1])
# print testy

# print data_sales_sum
# select the data of ki
ads_consume = np.loadtxt("ads_consume.txt")
bad_comment = np.loadtxt("bad_comment.txt")
dis_comment = np.loadtxt("dis_comment.txt")
good_comment = np.loadtxt("good_comment.txt")
ads_consume = np.nan_to_num(ads_consume)
ads_consume_k6 = ads_consume[shopid_k6]
bad_comment_k6 = bad_comment[shopid_k6]
dis_comment_k6 = dis_comment[shopid_k6]
good_comment_k6 = good_comment[shopid_k6]
# concat the data
data_trainX = pd.DataFrame(columns=[0,1,2,3,4])
for i in range(0,len(data_k1)):
    df1 = pd.DataFrame(data_k1.iloc[i:i+1,1:10])
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
    # data_train = df1.T
    # print df1.T.columns
    data_trainX = pd.concat([data_trainX,df1.T]
                 ,ignore_index = 'true')
# print data_trainX

# get the tranx from the data_tranX
trainX = pd.DataFrame(columns=[0,1,2,3,4])
for i in range(0,len(data_k1)):
    df1 = pd.DataFrame(data_trainX.iloc[9*i:9*i+5,0:5])
    trainX = pd.concat([trainX,df1]
                    ,ignore_index = 'true')
print trainX

testX = pd.DataFrame(columns=[0,1,2,3,4])
for i in range(0,len(data_k1)):
    df1 = pd.DataFrame(data_trainX.iloc[9*i+5:9*i+6,0:5])
    testX = pd.concat([testX,df1]
                    ,ignore_index = 'true')
# print testX

# guiyihua
x = pd.concat([trainX,testX])
x = x.fillna(0)
scaler_x = MinMaxScaler(feature_range=(0, 1))
x = scaler_x.fit_transform(x.values)
trainx = x[0:len(data_k1)*5,0:]
testx = x[len(data_k1)*5:,0:]
print len(trainx)
print len(testx)

y = pd.concat([trainy,testy])
y = y.fillna(0)
scaler_y = MinMaxScaler(feature_range=(0, 1))
y = scaler_y.fit_transform(y.values)
trainy = y[0:len(data_k1)*5,0:]
testy = y[len(data_k1)*5:,0:]
print len(trainx)
print len(trainy)
print len(testx)
print len(testy)



trainx = np.reshape(trainx, (trainx.shape[0], 1, 5))
testx = np.reshape(testx, (testx.shape[0], 1, 5))
# print trainx
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(12, input_shape=(1, 5)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainx, trainy, epochs=500, batch_size=1, verbose=2)


testPredict = model.predict(testx)
testPredict = scaler_y.inverse_transform(testPredict)
testy = scaler_y.inverse_transform(testy)

# print testPredict[0:,0]
# print testy
plt.plot(testPredict[0:,0],'b')
plt.plot(testy[0:,0],'r')
plt.show()
testScore = math.sqrt(mean_squared_error(testy[0:,0], testPredict[0:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# # load the dataset
# dataframe = read_csv('huaruifourmon_30min.csv', usecols=[1], engine='python', skipfooter=3)
# dataset = dataframe.values
# #dataframetest = read_csv('test.csv', usecols=[1], engine='python', skipfooter=3)
# #datasetest = dataframetest.values
# # transform int to float
# dataset = dataset.astype('float32')
# #datasetest = datasetest.astype('float32')
# matfn = 'ercitranx.mat'
# ercitranx = sio.loadmat(matfn)
# matfm = 'ercitrany.mat'
# ercitrany = sio.loadmat(matfm)
# print ercitranx
# print ercitrany
# # plt.plot(dataset)
# # plt.show()
#
# #plt.plot(datasetest)
# #plt.show()
#
# # X is the number of passengers at a given time (t) and Y is the number of passengers at the next time (t + 1).
#
# # convert an array of values into a dataset matrix
# def create_dataset(dataset, look_back=1):
#     dataX, dataY = [], []
#     for i in range(len(dataset)-look_back):
#         a = dataset[i:(i+look_back), 0]
#         dataX.append(a)
#         dataY.append(dataset[i + look_back, 0])
#     return numpy.array(dataX), numpy.array(dataY)
#
# # fix random seed for reproducibility
# numpy.random.seed(7)
#
# # normalize the dataset
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)
# # plt.plot(dataset)
# # plt.show()
# #datasetest = scaler.fit_transform(datasetest)
#
# # split into train and test sets
# train_size = int(len(dataset) * 0.67)
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
#
# # use this function to prepare the train and test datasets for modeling
# look_back =4
# trainX, trainY = create_dataset(train, look_back)
# testX, testY = create_dataset(test, look_back)
# # newtestX, newtestY = create_dataset(datasetest, look_back)
#
# # reshape input to be [samples, time steps, features]
# trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
# testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
# #newtestX = numpy.reshape(newtestX,(newtestX.shape[0], newtestX.shape[1], 1))
# print trainX
# print trainY
# # create and fit the LSTM network
# model = Sequential()
# model.add(LSTM(4, input_shape=(4, 1)))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)
#
#
# # make predictions
# trainPredict = model.predict(trainX)
# testPredict = model.predict(testX)
# #newtestPredict = model.predict(newtestX)
# plt.plot(testY[0:50],'bo-',label='true-value',markersize=10)
# plt.plot(testPredict[0:50],'gv-',label='predict-value',markersize=10)
# fig1 = plt.figure(1)
# axes = plt.subplot(111)
# axes.set_yticks([0,1])
# axes.grid(True)
# plt.legend(loc="higher right")
# plt.ylabel('occupancy')
# plt.xlabel('time')
# # plt.plot(testPredict[0:50])
# plt.show()
# # invert predictions
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])
# #newtestPredict = scaler.inverse_transform(newtestPredict)
# #newtestY = scaler.inverse_transform([newtestY])
#
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))
# #newtestScore = math.sqrt(mean_squared_error(newtestY[0], newtestPredict[:,0]))
# #print('NewTest Score: %.2f RMSE' % (newtestScore))
#
# # shift train predictions for plotting
# trainPredictPlot = numpy.empty_like(dataset)
# trainPredictPlot[:, :] = numpy.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
#
# # shift test predictions for plotting
# testPredictPlot = numpy.empty_like(dataset)
# testPredictPlot[:, :] = numpy.nan
# testPredictPlot[len(trainPredict)+(look_back*2):len(dataset)+1, :] = testPredict
#
# # shift newtest predictions for plotting
# #newtestPredictPlot = numpy.empty_like(datasetest)
# #newtestPredictPlot[:,:] = numpy.nan
# #newtestPredictPlot[look_back:len(newtestPredict)+look_back, :] = newtestPredict
#
# # plot baseline and predictions
# # plt.plot(scaler.inverse_transform(dataset))
# # plt.plot(trainPredictPlot)
# # plt.plot(testPredictPlot)
# # plt.show()
#
# #plt.plot(scaler.inverse_transform(datasetest))
# #plt.plot(newtestPredictPlot)
# #plt.show()
#
