import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import csv
import scipy.stats as stats
import seaborn as sns
import random
from statsmodels.tsa.stattools import acovf, acf, pacf, adfuller

fi = 0.2
sigma_2 = 0.4
n = 1000
N = 100
XS = []
for i in range(N):
    X_0 = 0
    Z = np.random.normal(loc=0,scale=np.sqrt(sigma_2),size=n)
    X = [X_0]
    for i in range(1,n):
        X.append(fi*X[i-1]+Z[i])
    XS.append(X)

def autokowariancja(h,x):
     return (1/len(x))*np.sum((x[:len(x)-abs(h)]-np.mean(x))*(x[abs(h):]-np.mean(x)))

def autokorelacja(h,x):
    return autokowariancja(h,x)/autokowariancja(0,x)

hs = np.linspace(0,50,51)

Kow_emp = []
Kor_emp = []
iter = 0
kor_teo = []
kow_teo = []
for xs in XS:

    kow_emp = []
    kor_emp = []
    for h in hs:
        kow_emp.append(autokowariancja(int(h),xs))
        kor_emp.append(acf(nlags=int(h),x=xs)[int(h)])
        if iter == 0:
            print(autokorelacja(int(h),xs))
            kor_teo.append(fi**abs(h))
            kow_teo.append(sigma_2*fi**abs(h)/(1-fi**2))
    Kor_emp.append(kor_emp)
    Kow_emp.append(kow_emp)
    iter+=1

lacvf = np.percentile(Kow_emp, 5, axis=0)
uacvf = np.percentile(Kow_emp, 95, axis=0)

lacf = np.percentile(Kor_emp, 5, axis=0)
uacf = np.percentile(Kor_emp, 95, axis=0)

plt.plot(hs,Kor_emp[0])
plt.scatter(hs,kor_teo,color="red")
plt.plot(hs,lacf,color="pink")
plt.plot(hs,uacf,color="pink")
plt.title("korelacja")
plt.show()


plt.plot(hs,Kow_emp[0])
plt.scatter(hs,kow_teo,color="red")
plt.plot(hs,lacvf,color="pink")
plt.plot(hs,uacvf,color="pink")
plt.title("kowariancja")
plt.show()

