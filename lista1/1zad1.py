import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import csv
x = []
y = []
c = np.loadtxt('/Users/wojtek/Desktop/KASC/1zad1.txt')
for p in c:
    x.append(p[0])
    y.append(p[1])
plt.scatter(x, y)
y_z_daszkiem = []
y_z_daszkiem_2 = []
for daszki in range(5):
    z = np.polyfit(x, y, daszki+1)
    y_z_daszkiem.append(np.polyval(z,sorted(x)))
    y_z_daszkiem_2.append(np.polyval(z,x))
bezwzgledne = []
kwadratowe = []
determinacja = []
for d in range(len(y_z_daszkiem_2)):
    plt.plot(sorted(x),y_z_daszkiem[d],label=f'stopień {d+1}')
    suma = sum(abs(y-y_z_daszkiem_2[d]))
    suma_kw = sum((y-y_z_daszkiem_2[d])**2)
    bezwzgledne.append(1/1000*suma)
    kwadratowe.append((1/1000)*suma_kw)
    determinacja.append(sum((y_z_daszkiem_2[d]-sum(y))**2)/sum((y-sum(y))**2))
plt.legend()
plt.show()
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.plot(np.linspace(1,5,5),bezwzgledne)
plt.subplot(1, 3, 2)
plt.plot(np.linspace(1,5,5),kwadratowe)
plt.subplot(1, 3, 3)
plt.plot(np.linspace(1,5,5),determinacja)
plt.show()