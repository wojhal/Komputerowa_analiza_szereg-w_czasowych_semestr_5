import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import csv
import scipy.stats as stats
import seaborn as sns
import random

n = 1000
sigma = 1
theta = 2
p  = np.arange(0.01,0.16,0.01)

a = np.arange(1,11,1)

M = 100

def autokowariancja(h,x):
    suma = 0
    for c in range(len(x)-abs(h)):
        suma += (x[c]-np.mean(x))*(x[c+abs(h)]-np.mean(x))
    return (1/len(x))*suma

def autokorelacja(h,x):
    return autokowariancja(h,x)/autokowariancja(0,x)

def ro(h,x):
    return 1/(len(x)-h) * sum(np.sign((x[i] - np.median(x)) * (x[i + abs(h)] - np.median(x))) for i in range(len(x)-abs(h)))

def costam(n,sigma,theta,a,p):
    Z = stats.norm.rvs(loc=0,scale=sigma,size=n+1)
    Y = []
    for ps in range(1,n+1):
        Y.append(Z[ps] + theta*Z[ps-1])
    szum = []
    for r in range(n):
        prob = np.random.random()
        if prob < p/2:
            szum.append(a)
        elif p/2 < prob < p:
            szum.append(-a)
        else: 
            szum.append(0)
    X = []
    for x in range(len(Y)):
        X.append(szum[x]+Y[x])
    return Y,X
# ts = np.linspace(1,n,n)
# print(ts)
# plt.plot(ts,Y)
# plt.show()
# plt.plot(ts,X)
# plt.show()

# hs = np.linspace(0,10,11)
# kor_emp_x = []
# kor_inne_x = []
# kor_emp_y = []
# kor_inne_y = []
# sum1, sum2 = 0, 0
# for h in hs:
#     kor_emp_y=autokorelacja(int(h),Y)
#     kor_inne_y=np.sin(np.pi/2*ro(int(h),Y))
#     kor_emp_x=autokorelacja(int(h),X)
#     kor_inne_x=np.sin(np.pi/2*ro(int(h),X))
# kor_teo = np.zeros(11)
# kor_teo[0],kor_teo[1] = 1,theta/(1+theta**2)

# plt.scatter(hs,kor_teo,color='red')
# plt.plot(hs,kor_emp_y)
# plt.plot(hs,kor_inne_y)
# plt.show()
# plt.scatter(hs,kor_teo,color='red')
# plt.plot(hs,kor_emp_x)
# plt.plot(hs,kor_inne_x)
# plt.show()

from joblib import Parallel, delayed  

def compute_sums(a, p, M=10,n=1000 ,sigma=1,theta=2):
    kor_teo = np.zeros(1)
    kor_teo[0]=theta/(1+theta**2)
    e1,e2 = 0, 0
    for i in range(M):
        print(i,a,p)
        Y,X = costam(n,sigma,theta,a,p)
        kor_emp=autokorelacja(1,X)
        kor_inne=ro(1,X)
        e1 += abs(kor_teo - kor_emp)
        e2 += abs(kor_teo - kor_inne)
    return e1 / M, e2 / M

results = Parallel(n_jobs=-1)(
    delayed(compute_sums)(ass, pss) for ass in a for pss in p
)
e1_a = np.array([res[0] for res in results]).reshape(len(a), len(p))

e2_a = np.array([res[1] for res in results]).reshape(len(a), len(p))

plt.figure(figsize=(12, 6))
sns.heatmap(e1_a, annot=False, cmap="viridis", xticklabels=p.round(2), yticklabels=a.round(2),vmin=0.1,vmax=0.3)
plt.xlabel("p")
plt.ylabel("a")
plt.title("Heatmapa dla e1_a")
plt.gca().invert_yaxis() 
plt.show()
plt.figure(figsize=(12, 6))
sns.heatmap(e2_a, annot=False, cmap="viridis", xticklabels=p.round(2), yticklabels=a.round(2),vmin=0.1,vmax=0.3)
plt.xlabel("p")
plt.ylabel("a")
plt.title("Heatmapa dla e2_a")
plt.gca().invert_yaxis() 
plt.show()














