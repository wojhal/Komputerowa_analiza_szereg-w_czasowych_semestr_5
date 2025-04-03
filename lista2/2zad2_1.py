import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import csv
import scipy.stats as stats

m = 5000
b1 = 5
def beta1(x,y):
    return np.sum(x*y)/np.sum(x**2)

ns = np.linspace(50,1000,20)
sigma = 1
b1_sr_teor_duza = []
b1_war_teor_duza = []
b1_sr_probka_duza = []
b1_war_probka_duza = []
for ns_ in ns:
    b_jedens = []
    X = np.linspace(0,10,int(ns_))
    for i in range(m):
        blad = stats.norm.rvs(loc=0, scale=sigma, size=int(ns_))
        y_nowe = b1*X + blad
        b1jeden = beta1(X,y_nowe)
        b_jedens.append(b1jeden)

    b1_sr_probka = np.mean(b_jedens)
    b1_sr_probka_duza.append(b1_sr_probka)
    b1_war_probka = np.var(b_jedens)
    b1_war_probka_duza.append(b1_war_probka)
    b1_sr_teor = b1
    b1_sr_teor_duza.append(b1_sr_teor)
    b1_war_teor = (sigma**2)*sum(X**2)/(sum(X**2))**2
    b1_war_teor_duza.append(b1_war_teor)

plt.plot(ns,b1_sr_probka_duza)
plt.plot(ns,b1_sr_teor_duza)
plt.ylim(4.999,5.001)
plt.title('B1 średnia')
plt.show()

plt.plot(ns,b1_war_probka_duza)
plt.plot(ns,b1_war_teor_duza)
plt.title('B1 wariancja')
plt.show()
