import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import csv
import scipy.stats as stats
import seaborn as sns
import random


fi = 0.2
sigma_2 = 0.4
n = 1000
X_0 = 0
Z = np.random.normal(loc=0,scale=np.sqrt(sigma_2),size=n)
hs = np.linspace(0,50,51)
X = [X_0]
for i in range(1,n):
    X.append(fi*X[i-1]+Z[i])


def autokowariancja(h,x):
     return (1/len(x))*np.sum((x[:len(x)-abs(h)]-np.mean(x))*(x[abs(h):]-np.mean(x)))

def autokorelacja(h,x):
    return autokowariancja(h,x)/autokowariancja(0,x)
kow_emp = []
kor_emp = []
kow_teo = []
kor_teo = []
for h in hs:
    kow_emp.append(autokowariancja(int(h),X))
    kor_emp.append(autokorelacja(int(h),X))
    kor_teo.append(fi**abs(h))
    kow_teo.append(sigma_2*fi**abs(h)/(1-fi**2))


plt.plot(hs,kor_emp)
plt.scatter(hs,kor_teo,color='red')
plt.title("korelacja")
plt.show()


plt.plot(hs,kow_emp)
plt.scatter(hs,kow_teo,color='red')
plt.title("kowariancja")
plt.show()

