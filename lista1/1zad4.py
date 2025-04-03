import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import csv

x = []
y = []
c = np.loadtxt('/Users/wojtek/Desktop/KASC/1zad4.txt')
for p in c:
    x.append(p[0])
    y.append(p[1])
#plt.scatter(x, y)
#plt.show()
def prosta_regresji(x,y):
    b_1 = np.sum(x*(y-np.mean(y)))/np.sum((x-np.mean(x))**2)
    b_0 = np.mean(y) - b_1 * np.mean(x)
    return b_0, b_1
b0, b1 = prosta_regresji(x, y)
xs = np.linspace(-20,20,1000)
regresja = b1*xs + b0
plt.scatter(x, y)
plt.plot(xs, regresja,color='orange')
plt.show()

y_z_daszkiem = []
for i in x:
    y_z_daszkiem.append(b1*i + b0)
residencia = []
for j in range(len(y)):
    residencia.append(y[j]-y_z_daszkiem[j])
xss = np.linspace(1,len(y),len(y))
plt.scatter(xss,residencia)
plt.show()


C = b0
A = b1
B = -1
odleglosci = []
zapasowa_x = []
zapasowa_y = []
for i in range(len(x)):
    odleglosci.append(abs(A*x[i]+B*y[i]+b0)/np.sqrt(A**2 + B**2))
for odl in range(len(odleglosci)):
    if odleglosci[odl] > 2:
        plt.scatter(x[odl], y[odl],color='black')
    else:
        zapasowa_x.append(x[odl])
        zapasowa_y.append(y[odl])
        plt.scatter(x[odl], y[odl],color='blue')
plt.plot(xs, regresja,color='orange')
plt.show()

b00, b11 = prosta_regresji(zapasowa_x, zapasowa_y)
plt.scatter(zapasowa_x,zapasowa_y)
regresja2 = b11*xs + b00
plt.plot(xs, regresja2,color='orange')
plt.show()