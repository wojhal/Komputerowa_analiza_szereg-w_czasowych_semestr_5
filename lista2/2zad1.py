import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import csv
import scipy.stats as stats

m = 5000
n = 1000
b0, b1 = 5, 2
ni = 5

def prosta_regresji(x,y):
    b_1 = np.sum(x*(y-np.mean(y)))/np.sum((x-np.mean(x))**2)
    b_0 = np.mean(y) - b_1 * np.mean(x)
    return b_0, b_1

X = np.linspace(0,10,1000)

bledy = stats.t.rvs(ni, loc=0, scale=1, size=n)
Y = b0 + b1*X + bledy
b00, b11 = prosta_regresji(X,Y)
plt.scatter(X,Y)
plt.plot(X,b00+b11*X,color='orange',linewidth=2)
plt.show()

rozne_n = np.linspace(100, 100000, 1000)
rozne_ni = np.linspace(0.01,100,1000)

def b1_estim(x,y):
    return np.sum((x - np.mean(x)) * (y))/np.sum((x-np.mean(x))**2)
def b0_estim(x,y):
    return np.mean(y) - b1_estim(x,y) * np.mean(x)

nis = np.linspace(5,100,20)
b0_sr_teor_duza = []
b1_sr_teor_duza = []
b0_war_teor_duza = []
b1_war_teor_duza = []
b0_sr_probka_duza = []
b1_sr_probka_duza = []
b0_war_probka_duza = []
b1_war_probka_duza = []
for nis_ in nis:
    b_zeros = []
    b_jedens = []
    for i in range(m):
        blad = stats.t.rvs(nis_, loc=0, scale=1, size=n)
        y_nowe = b0 + b1*X + blad
        b0zero, b1jeden = prosta_regresji(X,y_nowe)
        b_zeros.append(b0zero)
        b_jedens.append(b1jeden)

    b0_sr_probka = np.mean(b_zeros)
    b0_sr_probka_duza.append(b0_sr_probka)
    b1_sr_probka = np.mean(b_jedens)
    b1_sr_probka_duza.append(b1_sr_probka)
    b0_war_probka = np.var(b_zeros)
    b0_war_probka_duza.append(b0_war_probka)
    b1_war_probka = np.var(b_jedens)
    b1_war_probka_duza.append(b1_war_probka)
    b0_sr_teor = b0
    b0_sr_teor_duza.append(b0_sr_teor)
    b1_sr_teor = b1
    b1_sr_teor_duza.append(b1_sr_teor)
    b0_war_teor = ((1/n)+((np.mean(X)**2)/sum((x-np.mean(X))**2 for x in X)))*(nis_/(nis_-2))
    b0_war_teor_duza.append(b0_war_teor)
    b1_war_teor = (1/sum((x-np.mean(X))**2 for x in X))*(nis_/(nis_-2))
    b1_war_teor_duza.append(b1_war_teor)

plt.plot(nis,b0_sr_probka_duza)
plt.plot(nis,b0_sr_teor_duza)
plt.title('B0 średnia')
plt.show()

plt.plot(nis,b0_war_probka_duza)
plt.plot(nis,b0_war_teor_duza)
plt.title('B0 wariancja')
plt.show()

plt.plot(nis,b1_sr_probka_duza)
plt.plot(nis,b1_sr_teor_duza)
plt.title('B1 średnia')
plt.show()

plt.plot(nis,b1_war_probka_duza)
plt.plot(nis,b1_war_teor_duza)
plt.title('B1 wariancja')
plt.show()


    