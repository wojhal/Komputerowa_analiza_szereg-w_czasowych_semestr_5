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
w0s = []
w1s = []
for i in range(m):
    blad = stats.norm.rvs(loc=0, scale=sigma, size=m)
    y_nowe =b0 + b1*X + blad
    b0nowe, b1nowe = prosta_regresji(X,y_nowe)
    y_z_daszkiem = b1nowe*X+b0nowe
    s = np.sqrt(1/(m-2)*np.sum((y_nowe-y_z_daszkiem)**2))
    SE_beta_0 = s * np.sqrt((1/n + np.mean(X)**2/np.sum((X-np.mean(X))**2)))
    SE_beta_1 = s* np.sqrt(1/np.sum((X-np.mean(X))**2))
    W0 = (b0nowe - b0 )/ SE_beta_0 
    W1 = (b1nowe - b1) /  SE_beta_1
    w0s.append(W0)
    w1s.append(W1)

# x_1 = np.linspace(4.8,5.2,1000)
# x_0 = np.linspace(1.6, 2.4, 1000)
x_dens3 = np.linspace(-3,3,1000)
x_cdf1 = np.linspace(-3, 4, 1000)

sns.ecdfplot(w0s)
plt.plot(x_cdf1, stats.t.cdf(x_cdf1,m-2))
plt.show()


sns.ecdfplot(w1s)
plt.plot(x_cdf1, stats.t.cdf(x_cdf1,m-2))
plt.show()


sns.kdeplot(w0s)
plt.plot(x_cdf1, stats.t.pdf(x_cdf1,m-2))
plt.show()


sns.kdeplot(w1s)
plt.plot(x_cdf1, stats.t.pdf(x_cdf1,m-2))
plt.show()

print(stats.kstest(w0s,stats.t.rvs(m-2, size = m)))
print(stats.kstest(w1s,stats.t.rvs(m-2, size = m)))