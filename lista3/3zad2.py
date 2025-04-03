import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import csv
import scipy.stats as stats

b0, b1 = 2, 5
alfa = 0.05
sigma = 0.5
M = 1000


def pu_b0_unknown_var(alpha, x, y, b1_hat, b0_hat):
    n = len(x)
    residuals = y - (b1_hat * x + b0_hat)
    s_squared = np.sum(residuals**2) / (n - 2)
    s = np.sqrt(s_squared)
    return stats.t.ppf(1 - alpha/2, n - 2) * s * np.sqrt(1/n + np.mean(x)**2 / np.sum((x - np.mean(x))**2))


def pu_b1_unknown_var(alpha, x, y, b1_hat):
    n = len(x)
    residuals = y - (b1_hat * x + np.mean(y) - b1_hat * np.mean(x))
    s_squared = np.sum(residuals**2) / (n - 2)  
    s = np.sqrt(s_squared)
    return stats.t.ppf(1 - alpha/2, n - 2) * s / np.sqrt(np.sum((x - np.mean(x))**2))



def prosta_regresji(x,y):
    b_1 = np.sum(x*(y-np.mean(y)))/np.sum((x-np.mean(x))**2)
    b_0 = np.mean(y) - b_1 * np.mean(x)
    return b_0, b_1

xs = np.linspace(10,100,19)
dla_kazdego_x0 = []
dla_kazdego_x1 = []
nzn_dla_kazdego_x0 = []
nzn_dla_kazdego_x1 = []
dlugosci_0 = []
dlugosci_1 = []
nzn_dlugosci_0 = []
nzn_dlugosci_1 = []

for xs1 in xs:
    X = np.linspace(1,int(xs1),int(xs1))
    licznik0 = 0
    licznik1 = 0
    nznlicznik0 = 0
    nznlicznik1 = 0
    dlugosc_0 = 0
    dlugosc_1 = 0
    n_dlugosc_0 = 0
    n_dlugosc_1 = 0
    for m in range(M):
        bledy = stats.norm.rvs(loc=0, scale=sigma, size=int(xs1))
        Y = b0 + b1*X + bledy
        b00, b11 = prosta_regresji(X,Y)
        przedzialb1 = [-1*stats.norm.ppf(1-alfa/2) * sigma / np.sqrt(np.sum((X-np.mean(X))**2)) ,stats.norm.ppf(1-alfa/2) * sigma / np.sqrt(np.sum((X-np.mean(X))**2))]
        przedzialb0 = [-1*stats.norm.ppf(1 - alfa/2) * sigma * np.sqrt((1/ len(X) + np.mean(X)**2 / np.sum((X-np.mean(X))**2))) ,stats.norm.ppf(1 - alfa/2) * sigma * np.sqrt((1/ len(X) + np.mean(X)**2 / np.sum((X-np.mean(X))**2)))]
        nznprzedzialb1 = [-1*pu_b1_unknown_var(alfa, X, Y, b11),pu_b1_unknown_var(alfa, X, Y, b11)]
        nznprzedzialb0 = [-1*pu_b0_unknown_var(alfa, X, Y, b11, b00),pu_b0_unknown_var(alfa, X, Y, b11, b00)]
        if b00 + przedzialb0[0] <= b0 <= b00 + przedzialb0[1]:
            licznik0 += 1
        if  b11 + przedzialb1[0] <= b1 <= b11 + przedzialb1[1]:
            licznik1 += 1
        if b00 + nznprzedzialb0[0] <= b0 <= b00 + nznprzedzialb0[1]:
            nznlicznik0 += 1
        if b11 + nznprzedzialb1[0] <= b1 <= b11 + nznprzedzialb1[1]:
            nznlicznik1 += 1
        dlugosc_0 += 2*przedzialb0[1]
        dlugosc_1 += 2*przedzialb1[1]
        n_dlugosc_0 += 2*nznprzedzialb0[1]
        n_dlugosc_1 += 2*nznprzedzialb1[1]
    dla_kazdego_x0.append((licznik0/M)*100)
    dla_kazdego_x1.append((licznik1/M)*100)
    dlugosci_0.append(dlugosc_0/M)
    dlugosci_1.append(dlugosc_1/M)
    nzn_dlugosci_0.append(n_dlugosc_0/M)
    nzn_dlugosci_1.append(n_dlugosc_1/M)
    nzn_dla_kazdego_x0.append((nznlicznik0/M)*100)
    nzn_dla_kazdego_x1.append((nznlicznik1/M)*100)
plt.plot(xs,dla_kazdego_x0)
plt.plot(xs,nzn_dla_kazdego_x0)
plt.axhline(y=95,color='red')
plt.show()
plt.plot(xs,dla_kazdego_x1)
plt.plot(xs,nzn_dla_kazdego_x1)
plt.axhline(y=95,color='red')
plt.show()

plt.plot(xs,dlugosci_0)
plt.plot(xs,nzn_dlugosci_0)
plt.show()
plt.plot(xs,dlugosci_1)
plt.plot(xs,nzn_dlugosci_1)
plt.show()
