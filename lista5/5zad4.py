import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import csv
import scipy.stats as stats
import seaborn as sns
import random
from scipy.optimize import curve_fit

t = np.linspace(0,10,1001)
a1,a2 = 1.5,5
b1,b2 = 2,3
m_t = a1*t+a2
sigma, theta = 0.5,1
def s_t(t,b1,b2):
    return b1 * np.sin(b2 * t)
def ma1_sample(n, sigma, theta):
    z = np.random.normal(0, sigma, n+1)
    x = np.zeros(n)
    for i in range(1,n+1):
        x[i-1] = z[i] + theta * z[i-1]
    return x
X = ma1_sample(1001, 0.5, 1)
Y = m_t + s_t(t,b1,b2) + X
a1_daszek, a2_daszek = np.polyfit(t,Y, 1)
y_daszek = a1_daszek * t + a2_daszek
plt.plot(t, Y)
plt.plot(t, y_daszek)
plt.show()
Y_gwiazdka = Y - y_daszek

b1_daszek, b2_daszek = curve_fit(s_t, t, Y_gwiazdka,[max(Y_gwiazdka), 2*np.pi/2])[0]
y2_daszek = []
for t1 in t:
    y2_daszek.append(b1_daszek * np.sin(b2_daszek * t1))
plt.plot(t, Y_gwiazdka)
plt.plot(t, y2_daszek)
plt.show()
Y_dwie_gwiazdki = Y_gwiazdka - y2_daszek
plt.plot(t, Y_dwie_gwiazdki)
plt.show()
def autokowariancja(h,x):
     return (1/len(x))*np.sum((x[:len(x)-abs(h)]-np.mean(x))*(x[abs(h):]-np.mean(x)))

def autokorelacja(h,x):
    return autokowariancja(h,x)/autokowariancja(0,x)

hs = np.linspace(0,10,11)
kor_emp = []
for h in hs:
    kor_emp.append(autokorelacja(int(h), Y_dwie_gwiazdki))
kor_teo = np.zeros(11)
kor_teo[0],kor_teo[1]=1,theta/(1+theta**2)
plt.plot(hs, kor_teo, "ro")
plt.plot(hs, kor_emp)
plt.show()


