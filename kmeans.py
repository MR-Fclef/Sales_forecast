from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#read t_order_new.txt
k=40
j=[0]*k
t_order = np.loadtxt("t_order_new.txt")
nan_where = np.isnan(t_order)
t_order[nan_where] = 0
print t_order
kmeans = KMeans(n_clusters=k)
my_kmeans = kmeans.fit(t_order)
centroids = kmeans.labels_
inertia = kmeans.inertia_
print centroids
print len(centroids)
numSamples = len(t_order)
# mark1 = ['-r','-b','-g','-k','-c','-d','-f','-l','-y']
# for i in range(0,3000):
#     plt.plot(t_order[i],mark1[kmeans.labels_[i]])
# # mark2 = ['or','ob','og','ok','oc','om','of','ol','oy']
# # for i in range(k):
# #     plt.plot(centroids[i],mark2[i],markersize=12)
# plt.show()
print centroids.tolist().count(1)
k1=[[0 for col in range(10)] for row in range(centroids.tolist().count(0))]
k2=[[0 for col in range(10)] for row in range(centroids.tolist().count(1))]
k3=[[0 for col in range(10)] for row in range(centroids.tolist().count(2))]
k4=[[0 for col in range(10)] for row in range(centroids.tolist().count(3))]
k5=[[0 for col in range(10)] for row in range(centroids.tolist().count(4))]
k6=[[0 for col in range(10)] for row in range(centroids.tolist().count(5))]
k7=[[0 for col in range(10)] for row in range(centroids.tolist().count(6))]
k8=[[0 for col in range(10)] for row in range(centroids.tolist().count(7))]
k9=[[0 for col in range(10)] for row in range(centroids.tolist().count(8))]
k10=[[0 for col in range(10)] for row in range(centroids.tolist().count(9))]
k11=[[0 for col in range(10)] for row in range(centroids.tolist().count(10))]
k12=[[0 for col in range(10)] for row in range(centroids.tolist().count(11))]
k13=[[0 for col in range(10)] for row in range(centroids.tolist().count(12))]
k14=[[0 for col in range(10)] for row in range(centroids.tolist().count(13))]
k15=[[0 for col in range(10)] for row in range(centroids.tolist().count(14))]
k16=[[0 for col in range(10)] for row in range(centroids.tolist().count(15))]
k17=[[0 for col in range(10)] for row in range(centroids.tolist().count(16))]
k18=[[0 for col in range(10)] for row in range(centroids.tolist().count(17))]
k19=[[0 for col in range(10)] for row in range(centroids.tolist().count(18))]
k20=[[0 for col in range(10)] for row in range(centroids.tolist().count(19))]
k21=[[0 for col in range(10)] for row in range(centroids.tolist().count(20))]
k22=[[0 for col in range(10)] for row in range(centroids.tolist().count(21))]
k23=[[0 for col in range(10)] for row in range(centroids.tolist().count(22))]
k24=[[0 for col in range(10)] for row in range(centroids.tolist().count(23))]
k25=[[0 for col in range(10)] for row in range(centroids.tolist().count(24))]
k26=[[0 for col in range(10)] for row in range(centroids.tolist().count(25))]
k27=[[0 for col in range(10)] for row in range(centroids.tolist().count(26))]
k28=[[0 for col in range(10)] for row in range(centroids.tolist().count(27))]
k29=[[0 for col in range(10)] for row in range(centroids.tolist().count(28))]
k30=[[0 for col in range(10)] for row in range(centroids.tolist().count(29))]
k31=[[0 for col in range(10)] for row in range(centroids.tolist().count(30))]
k32=[[0 for col in range(10)] for row in range(centroids.tolist().count(31))]
k33=[[0 for col in range(10)] for row in range(centroids.tolist().count(32))]
k34=[[0 for col in range(10)] for row in range(centroids.tolist().count(33))]
k35=[[0 for col in range(10)] for row in range(centroids.tolist().count(34))]
k36=[[0 for col in range(10)] for row in range(centroids.tolist().count(35))]
k37=[[0 for col in range(10)] for row in range(centroids.tolist().count(36))]
k38=[[0 for col in range(10)] for row in range(centroids.tolist().count(37))]
k39=[[0 for col in range(10)] for row in range(centroids.tolist().count(38))]
k40=[[0 for col in range(10)] for row in range(centroids.tolist().count(39))]
#
for i in range(0,3000):
    if centroids[i]==0:
        k1[j[centroids[i]]][0:9]=t_order[i]
        k1[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==1:
        k2[j[centroids[i]]][0:9]=t_order[i]
        k2[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==2:
        k3[j[centroids[i]]][0:9]=t_order[i]
        k3[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==3:
        k4[j[centroids[i]]][0:9]=t_order[i]
        k4[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==4:
        k5[j[centroids[i]]][0:9]=t_order[i]
        k5[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==5:
        k6[j[centroids[i]]][0:9]=t_order[i]
        k6[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==6:
        k7[j[centroids[i]]][0:9]=t_order[i]
        k7[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==7:
        k8[j[centroids[i]]][0:9]=t_order[i]
        k8[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==8:
        k9[j[centroids[i]]][0:9]=t_order[i]
        k9[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==9:
        k10[j[centroids[i]]][0:9]=t_order[i]
        k10[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==10:
        k11[j[centroids[i]]][0:9]=t_order[i]
        k11[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==11:
        k12[j[centroids[i]]][0:9]=t_order[i]
        k12[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==12:
        k13[j[centroids[i]]][0:9]=t_order[i]
        k13[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==13:
        k14[j[centroids[i]]][0:9]=t_order[i]
        k14[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==14:
        k15[j[centroids[i]]][0:9]=t_order[i]
        k15[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==15:
        k16[j[centroids[i]]][0:9]=t_order[i]
        k16[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==16:
        k17[j[centroids[i]]][0:9]=t_order[i]
        k17[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==17:
        k18[j[centroids[i]]][0:9]=t_order[i]
        k18[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==18:
        k19[j[centroids[i]]][0:9]=t_order[i]
        k19[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==19:
        k20[j[centroids[i]]][0:9]=t_order[i]
        k20[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==20:
        k21[j[centroids[i]]][0:9]=t_order[i]
        k21[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==21:
        k22[j[centroids[i]]][0:9]=t_order[i]
        k22[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==22:
        k23[j[centroids[i]]][0:9]=t_order[i]
        k23[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==23:
        k24[j[centroids[i]]][0:9]=t_order[i]
        k24[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==24:
        k25[j[centroids[i]]][0:9]=t_order[i]
        k25[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==25:
        k26[j[centroids[i]]][0:9]=t_order[i]
        k26[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==26:
        k27[j[centroids[i]]][0:9]=t_order[i]
        k27[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==27:
        k28[j[centroids[i]]][0:9]=t_order[i]
        k28[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==28:
        k29[j[centroids[i]]][0:9]=t_order[i]
        k29[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==29:
        k30[j[centroids[i]]][0:9]=t_order[i]
        k30[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==30:
        k31[j[centroids[i]]][0:9]=t_order[i]
        k31[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==31:
        k32[j[centroids[i]]][0:9]=t_order[i]
        k32[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==32:
        k33[j[centroids[i]]][0:9]=t_order[i]
        k33[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==33:
        k34[j[centroids[i]]][0:9]=t_order[i]
        k34[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==34:
        k35[j[centroids[i]]][0:9]=t_order[i]
        k35[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==35:
        k36[j[centroids[i]]][0:9]=t_order[i]
        k36[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==36:
        k37[j[centroids[i]]][0:9]=t_order[i]
        k37[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==37:
        k38[j[centroids[i]]][0:9]=t_order[i]
        k38[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==38:
        k39[j[centroids[i]]][0:9]=t_order[i]
        k39[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    if centroids[i]==39:
        k40[j[centroids[i]]][0:9]=t_order[i]
        k40[j[centroids[i]]][9]=i
        j[centroids[i]]=j[centroids[i]]+1
    # if centroids[i]==40:
    #     k1[j[centroids[i]]][0:9]=t_order[i]
    #     k1[j[centroids[i]]][9]=i
    #     j[centroids[i]]=j[centroids[i]]+1
print np.array(k1,dtype=int)
for i in range(0,len(np.array(k9,dtype=int))):
    plt.plot(np.array(k9,dtype=int)[i])
plt.show()
data_k1 = pd.DataFrame(np.array(k1,dtype=int))
data_k1.to_csv('data_k1.csv')
data_k2 = pd.DataFrame(np.array(k2,dtype=int))
data_k2.to_csv('data_k2.csv')
data_k3 = pd.DataFrame(np.array(k3,dtype=int))
data_k3.to_csv('data_k3.csv')
data_k4 = pd.DataFrame(np.array(k4,dtype=int))
data_k4.to_csv('data_k4.csv')
data_k5 = pd.DataFrame(np.array(k5,dtype=int))
data_k5.to_csv('data_k5.csv')
data_k6 = pd.DataFrame(np.array(k6,dtype=int))
data_k6.to_csv('data_k6.csv')
data_k7 = pd.DataFrame(np.array(k7,dtype=int))
data_k7.to_csv('data_k7.csv')
data_k8 = pd.DataFrame(np.array(k8,dtype=int))
data_k8.to_csv('data_k8.csv')
data_k9 = pd.DataFrame(np.array(k9,dtype=int))
data_k9.to_csv('data_k9.csv')
data_k10 = pd.DataFrame(np.array(k10,dtype=int))
data_k10.to_csv('data_k10.csv')
data_k11 = pd.DataFrame(np.array(k11,dtype=int))
data_k11.to_csv('data_k11.csv')
data_k12 = pd.DataFrame(np.array(k12,dtype=int))
data_k12.to_csv('data_k12.csv')
data_k13 = pd.DataFrame(np.array(k13,dtype=int))
data_k13.to_csv('data_k13.csv')
data_k14 = pd.DataFrame(np.array(k14,dtype=int))
data_k14.to_csv('data_k14.csv')
data_k15 = pd.DataFrame(np.array(k15,dtype=int))
data_k15.to_csv('data_k15.csv')

data_k16 = pd.DataFrame(np.array(k16,dtype=int))
data_k16.to_csv('data_k16.csv')
data_k17 = pd.DataFrame(np.array(k17,dtype=int))
data_k17.to_csv('data_k17.csv')
data_k18 = pd.DataFrame(np.array(k18,dtype=int))
data_k18.to_csv('data_k18.csv')
data_k19 = pd.DataFrame(np.array(k19,dtype=int))
data_k19.to_csv('data_k19.csv')
data_k20 = pd.DataFrame(np.array(k20,dtype=int))
data_k20.to_csv('data_k20.csv')
data_k21 = pd.DataFrame(np.array(k21,dtype=int))
data_k21.to_csv('data_k21.csv')
data_k22 = pd.DataFrame(np.array(k22,dtype=int))
data_k22.to_csv('data_k22.csv')
data_k23 = pd.DataFrame(np.array(k23,dtype=int))
data_k23.to_csv('data_k23.csv')
data_k24 = pd.DataFrame(np.array(k24,dtype=int))
data_k24.to_csv('data_k24.csv')
data_k25 = pd.DataFrame(np.array(k25,dtype=int))
data_k25.to_csv('data_k25.csv')
data_k26 = pd.DataFrame(np.array(k26,dtype=int))
data_k26.to_csv('data_k26.csv')
data_k27 = pd.DataFrame(np.array(k27,dtype=int))
data_k27.to_csv('data_k27.csv')
data_k28 = pd.DataFrame(np.array(k28,dtype=int))
data_k28.to_csv('data_k28.csv')
data_k29 = pd.DataFrame(np.array(k29,dtype=int))
data_k29.to_csv('data_k29.csv')
data_k30 = pd.DataFrame(np.array(k30,dtype=int))
data_k30.to_csv('data_k30.csv')
data_k31 = pd.DataFrame(np.array(k31,dtype=int))
data_k31.to_csv('data_k31.csv')
data_k32 = pd.DataFrame(np.array(k32,dtype=int))
data_k32.to_csv('data_k32.csv')
data_k33 = pd.DataFrame(np.array(k33,dtype=int))
data_k33.to_csv('data_k33.csv')
data_k34 = pd.DataFrame(np.array(k34,dtype=int))
data_k34.to_csv('data_k34.csv')
data_k35 = pd.DataFrame(np.array(k35,dtype=int))
data_k35.to_csv('data_k35.csv')
data_k36 = pd.DataFrame(np.array(k36,dtype=int))
data_k36.to_csv('data_k36.csv')
data_k37 = pd.DataFrame(np.array(k37,dtype=int))
data_k37.to_csv('data_k37.csv')
data_k38 = pd.DataFrame(np.array(k38,dtype=int))
data_k38.to_csv('data_k38.csv')
data_k39 = pd.DataFrame(np.array(k39,dtype=int))
data_k39.to_csv('data_k39.csv')
data_k40 = pd.DataFrame(np.array(k40,dtype=int))
data_k40.to_csv('data_k40.csv')

    # if centroids[i]==0:
    #     for j in range(0,centroids.tolist().count(0)):
    #         k1[j]=t_order[i]
    #         k1[j][9]=i


