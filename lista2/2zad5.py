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
sigma2 = 5
x1 = np.random.normal(0,np.sqrt(sigma1),400)
x2 = np.random.normal(0,np.sqrt(sigma2),600)
x = list(x1)
for xss in x2:
    x.append(xss)
xs = np.linspace(1,1000,1000).astype(int)
plt.plot(xs,x)
plt.show()
c = np.cumsum(np.array(x)**2)
plt.plot(xs,c)
plt.show()
k = 399
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
plt.plot(xs2,lista)
plt.show()
print(np.argmin(lista))

