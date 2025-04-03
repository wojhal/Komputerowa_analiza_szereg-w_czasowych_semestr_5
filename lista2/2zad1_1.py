import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import csv
import scipy.stats as stats

m = 5000
b0, b1 = 5, 2
ni = 10

def prosta_regresji(x,y):
    b_1 = np.sum(x*(y-np.mean(y)))/np.sum((x-np.mean(x))**2)
    b_0 = np.mean(y) - b_1 * np.mean(x)
    return b_0, b_1

def b1_estim(x,y):
    return np.sum((x - np.mean(x)) * (y))/np.sum((x-np.mean(x))**2)
def b0_estim(x,y):
    return np.mean(y) - b1_estim(x,y) * np.mean(x)

ns = np.linspace(50,1000,20)
b0_sr_teor_duza = []
b1_sr_teor_duza = []
b0_war_teor_duza = []
b1_war_teor_duza = []
b0_sr_probka_duza = []
b1_sr_probka_duza = []
b0_war_probka_duza = []
b1_war_probka_duza = []
for ns_ in ns:
    b_zeros = []
    b_jedens = []
    X = np.linspace(0,10,int(ns_))
    for i in range(m):
        blad = stats.t.rvs(ni, loc=0, scale=1, size=int(ns_))
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
    b0_war_teor = ((1/ns_)+((np.mean(X)**2)/sum((x-np.mean(X))**2 for x in X)))*(ni/(ni-2))
    b0_war_teor_duza.append(b0_war_teor)
    b1_war_teor = (1/sum((x-np.mean(X))**2 for x in X))*(ni/(ni-2))
    b1_war_teor_duza.append(b1_war_teor)
plt.plot(ns,b0_sr_probka_duza)
plt.plot(ns,b0_sr_teor_duza)
plt.title('B0 średnia')
plt.show()

plt.plot(ns,b0_war_probka_duza)
plt.plot(ns,b0_war_teor_duza)
plt.title('B0 wariancja')
plt.show()

plt.plot(ns,b1_sr_probka_duza)
plt.plot(ns,b1_sr_teor_duza)
plt.title('B1 średnia')
plt.show()

plt.plot(ns,b1_war_probka_duza)
plt.plot(ns,b1_war_teor_duza)
plt.title('B1 wariancja')
plt.show()