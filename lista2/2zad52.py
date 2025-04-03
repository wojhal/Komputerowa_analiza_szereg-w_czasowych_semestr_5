import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import csv
import scipy.stats as stats
import seaborn as sns

def prosta_regresji(x,y):
    b_1 = np.sum(x*(y-np.mean(y)))/np.sum((x-np.mean(x))**2)
    b_0 = np.mean(y) - b_1 * np.mean(x)
    return b_0, b_1


sigma1 = 1
sigmas2 = [1.5,3,4.5,6]
listaa = []
for sigma2 in sigmas2:
    rozne_listy = []
    for c in range(200):
        x1 = np.random.normal(0,np.sqrt(sigma1),50)
        x2 = np.random.normal(0,np.sqrt(sigma2),950)
        x = list(x1)
        for xss in x2:
            x.append(xss)
        xs = np.linspace(1,1000,1000).astype(int)

        c = np.cumsum(np.array(x)**2)

        k = 300
        c1, c2 = c[:k], c[k:]
        b0_1, b1_1 = prosta_regresji(xs[:k], c1)
        b0_2, b1_2 = prosta_regresji(xs[k:], c2)
        def y1(k):
            return b1_1*xs[:k] + b0_1
        def y2(k):
            return b1_2*xs[k:] + b0_2
        lista = []
        xs2 = np.linspace(1,998,998)
        for i in range(1,999):
            lista.append(np.sum((c[:i] - y1(i)) ** 2) + np.sum((c[i:] - y2(i)) ** 2))
        rozne_listy.append(np.argmin(lista))
    listaa.append(rozne_listy)

plt.axhline(y=50)
plt.boxplot(listaa, labels=['sigma 1.5', 'sigma 3', 'sigma 4.5', 'sigma 6'])
plt.plot()
plt.show()

