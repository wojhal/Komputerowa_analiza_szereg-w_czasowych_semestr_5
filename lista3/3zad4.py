import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import csv
import scipy.stats as stats

x = []
y = []
c = np.loadtxt('/Users/wojtek/Desktop/KASC/1zad4.txt')
c = sorted(c,key=lambda x: x[0])
for p in c:
    x.append(p[0])
    y.append(p[1])


treningowe_x = []
treningowe_y = []
testowe_x = []
testowe_y = []
for i in range(len(x)):
    if i < 990:
        treningowe_x.append(x[i])
        treningowe_y.append(y[i])
    else:
        testowe_x.append(x[i])
        testowe_y.append(y[i])


def prosta_regresji(x,y):
    b_1 = np.sum(x*(y-np.mean(y)))/np.sum((x-np.mean(x))**2)
    b_0 = np.mean(y) - b_1 * np.mean(x)
    return b_0, b_1

b0, b1 = prosta_regresji(treningowe_x, treningowe_y)
prognoza_punktowa = []
for f in x:
    prognoza_punktowa.append(b0 + b1*f)

prognoza_przedzialowa_1 = []
prognoza_przedzialowa_2 = []
iter = 990
s = (1/988)*np.sum((y[c]-prognoza_punktowa[c])**2 for c in range(len(prognoza_punktowa)-10))
for x_00 in testowe_x:
    prognoza_przedzialowa_1.append(prognoza_punktowa[iter]-stats.t.ppf(1-0.05/2, 988)*np.sqrt(s*(1+(1/990)+((x_00-np.mean(treningowe_x))**2)/np.sum((treningowe_x-np.mean(treningowe_x))**2) )))
    prognoza_przedzialowa_2.append(prognoza_punktowa[iter]+stats.t.ppf(1-0.05/2, 988)*np.sqrt(s*(1+(1/990)+((x_00-np.mean(treningowe_x))**2)/np.sum((xy-np.mean(treningowe_x))**2 for xy in treningowe_x))))
    iter+=1

print(prognoza_przedzialowa_1)
plt.scatter(treningowe_x,treningowe_y)
plt.scatter(testowe_x,testowe_y,color='red')
plt.plot(x,prognoza_punktowa,color='green')
plt.plot(testowe_x,prognoza_przedzialowa_1,color='pink')
plt.plot(testowe_x,prognoza_przedzialowa_2,color='pink')
plt.show()
