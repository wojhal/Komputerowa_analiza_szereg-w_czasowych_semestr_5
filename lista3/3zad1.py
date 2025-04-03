import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import csv
import scipy.stats as stats

b0, b1 = 2, 5
alfa = 0.05
sigmas = [0.01,0.5,1]
M = 1000
def b1_estim(x,y):
    return np.sum((x - np.mean(x)) * (y))/np.sum((x-np.mean(x))**2)
def b0_estim(x,y):
    return np.mean(y) - b1_estim(x,y) * np.mean(x)

def prosta_regresji(x,y):
    b_1 = np.sum(x*(y-np.mean(y)))/np.sum((x-np.mean(x))**2)
    b_0 = np.mean(y) - b_1 * np.mean(x)
    return b_0, b_1
xs = np.linspace(10,100,19)
sigmastyczne0 = []
sigmastyczne1 = []
for sigma in sigmas:
    dla_kazdego_x0 = []
    dla_kazdego_x1 = []
    for xs1 in xs:
        X = np.linspace(1,int(xs1),int(xs1))
        licznik0 = 0
        licznik1 = 0
        for m in range(M):
            bledy = stats.norm.rvs(loc=0, scale=sigma, size=int(xs1))
            Y = b0 + b1*X + bledy
            b00, b11 = prosta_regresji(X,Y)
            przedzialb1 = [-1*stats.norm.ppf(1-alfa/2) * sigma / np.sqrt(np.sum((X-np.mean(X))**2)) ,stats.norm.ppf(1-alfa/2) * sigma / np.sqrt(np.sum((X-np.mean(X))**2))]
            przedzialb0 = [-1*stats.norm.ppf(1 - alfa/2) * sigma * np.sqrt((1/ len(X) + np.mean(X)**2 / np.sum((X-np.mean(X))**2))) ,stats.norm.ppf(1 - alfa/2) * sigma * np.sqrt((1/ len(X) + np.mean(X)**2 / np.sum((X-np.mean(X))**2)))]
            if b00 + przedzialb0[0] <= b0 <= b00 + przedzialb0[1]:
                licznik0 += 1
            if b1 >= b11 + przedzialb1[0] and b1 <= b11 + przedzialb1[1]:
                licznik1 += 1
        dla_kazdego_x0.append((licznik0/M)*100)
        dla_kazdego_x1.append((licznik1/M)*100)
    sigmastyczne0.append(dla_kazdego_x0)
    sigmastyczne1.append(dla_kazdego_x1)
plt.plot(xs,sigmastyczne0[0])
plt.plot(xs,sigmastyczne0[1])
plt.plot(xs,sigmastyczne0[2])
plt.axhline(y=95,color='red')
plt.show()
plt.plot(xs,sigmastyczne1[0])
plt.plot(xs,sigmastyczne1[1])
plt.plot(xs,sigmastyczne1[2])
plt.axhline(y=95,color='red')
plt.show()