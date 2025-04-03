import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import csv

x = []
y = []
c = np.loadtxt('/Users/wojtek/Desktop/KASC/1zad6.txt')
for p in c:
    x.append(p[0])
    y.append(p[1])

def prosta_regresji(x,y):
    b_1 = np.sum(x*(y-np.mean(y)))/np.sum((x-np.mean(x))**2)
    b_0 = np.mean(y) - b_1 * np.mean(x)
    return b_0, b_1

plt.scatter(x,y)
plt.show()

y_transformed = np.log(y)
b0, b1 = prosta_regresji(x,y_transformed)
transformed_line = []
for c in x:
    transformed_line.append(b0+c*b1)
xs = np.linspace(-5,5,1000)
plt.scatter(x,y_transformed)
plt.plot(xs,b0+xs*b1,color='orange')
plt.show()
a,b = np.exp(b0), b1
plt.scatter(x,y)
plt.plot(xs,a*np.exp(b*xs),color='orange')
plt.show()
