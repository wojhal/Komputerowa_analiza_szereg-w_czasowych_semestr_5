import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import csv
from tabulate import tabulate

x = []
y = []
c = np.loadtxt('/Users/wojtek/Desktop/KASC/1zad4.txt')
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
xs = np.linspace(-20,20,1000)
regresja = b1*xs + b0
plt.scatter(treningowe_x, treningowe_y)
plt.plot(xs, regresja,color='orange')
plt.show()
#dane testowe
b00, b11 = prosta_regresji(testowe_x, testowe_y)
y_z_daszkiem = []
for i in testowe_x:
    y_z_daszkiem.append(b11*i + b00)
residencia = []
for j in range(len(testowe_y)):
    residencia.append(testowe_y[j]-y_z_daszkiem[j])
xss = np.linspace(1,len(testowe_y),len(testowe_y))
plt.scatter(xss,residencia)
plt.show()
y_z_daszkiem_tren = []
for i in treningowe_x:
    y_z_daszkiem_tren.append(b1*i + b0)
sr_b_res = []
sr_k_res = []
sr_b_tren = []
sr_k_tren = []
for f in range(len(testowe_y)):
    sr_b_res.append(abs(testowe_y[f]-y_z_daszkiem[f]))
    sr_k_res.append((testowe_y[f]-y_z_daszkiem[f])**2)
for d in range(len(treningowe_y)):
    sr_b_tren.append(abs(treningowe_y[d]-y_z_daszkiem_tren[d]))
    sr_k_tren.append((treningowe_y[d]-y_z_daszkiem_tren[d])**2)
srr_b_res = (1/len(testowe_y))*sum(sr_b_res)
srr_k_res = (1/len(testowe_y))*sum(sr_k_res)
srr_b_tren = (1/len(treningowe_y))*sum(sr_b_tren)
srr_k_tren = (1/len(treningowe_y))*sum(sr_k_tren)

head = ['Treningowe','Testowe']
data = [
    ['MSE', str(srr_k_tren),str(srr_k_res)],
    ['d', str(srr_b_tren),str(srr_b_res)],
]
print(tabulate(data,headers=head))


