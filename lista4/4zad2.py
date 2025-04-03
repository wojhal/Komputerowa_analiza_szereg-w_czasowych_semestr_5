import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import csv
import scipy.stats as stats
import seaborn as sns

theta = 2
sigma = 1
n = 1000


def autokowariancja(h,x):
    suma = 0
    for c in range(len(x)-abs(h)):
        suma += (x[c]-np.mean(x))*(x[c+abs(h)]-np.mean(x))
    return (1/len(x))*suma

def autokorelacja(h,x):
    return autokowariancja(h,x)/autokowariancja(0,x)

Z = stats.norm.rvs(loc=0,scale=sigma,size=n+1)
X = []
for p in range(n):
    X.append(Z[p+1] + theta*Z[p])
hs = np.linspace(0,10,11)
kow_emp = []
kor_emp = []
for h in hs:
    kow_emp.append(autokowariancja(int(h),X))
    kor_emp.append(autokorelacja(int(h),X))
kow_teo = np.zeros(11)
kow_teo[0],kow_teo[1] = (sigma**2)*(1+theta**2),theta*sigma**2

kor_teo = np.zeros(11)
kor_teo[0],kor_teo[1] = 1,theta/(1+theta**2)

plt.plot(hs,kow_emp)
plt.scatter(hs,kow_teo,color='red')
plt.show()

plt.plot(hs,kor_emp)
plt.scatter(hs,kor_teo,color='red')
plt.show()
