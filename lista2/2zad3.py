import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import csv
import scipy.stats as stats
import seaborn as sns

m = 1000
b0, b1 = 2, 5
sigma = 2
n = 1000
def prosta_regresji(x,y):
    b_1 = np.sum(x*(y-np.mean(y)))/np.sum((x-np.mean(x))**2)
    b_0 = np.mean(y) - b_1 * np.mean(x)
    return b_0, b_1

X = np.linspace(0,10,1000)
b0s = []
b1s = []
for i in range(m):
    blad = stats.norm.rvs(loc=0, scale=sigma, size=m)
    y_nowe =b0 + b1*X + blad
    b0nowe, b1nowe = prosta_regresji(X,y_nowe)
    b0s.append(b0nowe)
    b1s.append(b1nowe)
b0_teor = stats.norm.rvs(loc=b0, scale=(sigma**2)*((1/m)+((sum(X)**2)/(sum((x-sum(X))**2 for x in X)))), size=m)
b1_teor = stats.norm.rvs(loc=b1, scale=(sigma**2)*(1/sum((x-sum(X))**2 for x in X)), size=m)
plt.hist(b0s,density=True,bins=30)
plt.title('b0')
plt.show()

plt.hist(b1s,density=True,bins=30)
plt.title('b1')
plt.show()
x_1 = np.linspace(4.8,5.2,1000)
x_0 = np.linspace(1.6, 2.4, 1000)


sns.ecdfplot(b0s)
plt.plot(x_0,stats.norm.cdf(x_0,b0, np.sqrt(((1/m)+((np.mean(X)**2)/np.sum((X-np.mean(X))**2)))*(sigma**2))))
plt.show()


sns.ecdfplot(b1s)
plt.plot(x_1,stats.norm.cdf(x_1,b1, np.sqrt((1/np.sum((X-np.mean(X))**2))*(sigma**2))))
plt.show()


sns.kdeplot(b0s)
plt.plot(x_0,stats.norm.pdf(x_0,b0, np.sqrt(((1/m)+((np.mean(X)**2)/np.sum((X-np.mean(X))**2)))*(sigma**2))))
plt.show()


sns.kdeplot(b1s)
plt.plot(x_1,stats.norm.pdf(x_1,b1, np.sqrt((1/np.sum((X-np.mean(X))**2))*(sigma**2))))
plt.show()

print(stats.kstest(b0s,np.random.normal(b0,np.sqrt(((1/m)+((np.mean(X)**2)/np.sum((X-np.mean(X))**2)))*(sigma**2)),m)))
print(stats.kstest(b1s,np.random.normal(b1,np.sqrt((1/np.sum((X-np.mean(X))**2))*(sigma**2)),m)))