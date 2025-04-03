import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import csv
import scipy.stats as stats
import seaborn as sns

def autokowariancja(h,x):
    suma = 0
    for c in range(len(x)-abs(h)):
        suma += (x[c]-np.mean(x))*(x[c+abs(h)]-np.mean(x))
    return (1/len(x))*suma

def autokorelacja(h,x):
    return autokowariancja(h,x)/autokowariancja(0,x)

ns= np.linspace(20,2000,199)
kow_n = []
kor_n = []
for n in ns:
    print(n)
    X = stats.norm.rvs(loc=0,scale=2,size=int(n))
    hs = np.linspace(0,10,11)
    kow_emp = []
    kor_emp = []
    for h in hs:
        kow_emp.append(autokowariancja(int(h),X))
        kor_emp.append(autokorelacja(int(h),X))
    kow_teo = np.zeros(11)
    kow_teo[0] = 4
    kor_teo = np.zeros(11)
    kor_teo[0] = 1
    kow_e = (1/10) * np.sum(abs(kow_emp-kow_teo))
    kor_e = (1/10) * np.sum(abs(kor_emp-kor_teo))
    kow_n.append(kow_e)
    kor_n.append(kor_e)


plt.plot(ns,kow_n)
plt.show()
plt.plot(ns,kor_n)
plt.show()


# plt.plot(hs,kow_emp)
# plt.scatter(hs,kow_teo,color='red')
# plt.show()

# plt.plot(hs,kor_emp)
# plt.scatter(hs,kor_teo,color='red')
# plt.show()