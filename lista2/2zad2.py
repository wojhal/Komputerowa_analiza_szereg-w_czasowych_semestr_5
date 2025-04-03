import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import csv
import scipy.stats as stats

m = 5000
n = 1000
b1 = 5
def beta1(x,y):
    return np.sum(x*y)/np.sum(x**2)


X = np.linspace(0,10,1000)
sigma = np.linspace(1,50,50)
b1_sr_teor_duza = []
b1_war_teor_duza = []
b1_sr_probka_duza = []
b1_war_probka_duza = []
for sigms in sigma:
    b_jedens = []
    for i in range(m):
        blad = stats.norm.rvs(loc=0, scale=sigms, size=n)
        y_nowe = b1*X + blad
        b1jeden = beta1(X,y_nowe)
        b_jedens.append(b1jeden)

    b1_sr_probka = np.mean(b_jedens)
    b1_sr_probka_duza.append(b1_sr_probka)
    b1_war_probka = np.var(b_jedens)
    b1_war_probka_duza.append(b1_war_probka)
    b1_sr_teor = b1
    b1_sr_teor_duza.append(b1_sr_teor)
    b1_war_teor = (sigms**2)*sum(X**2)/(sum(X**2))**2
    b1_war_teor_duza.append(b1_war_teor)


plt.plot(sigma,b1_sr_probka_duza)
plt.plot(sigma,b1_sr_teor_duza)
plt.title('B1 średnia')
plt.show()

plt.plot(sigma,b1_war_probka_duza)
plt.plot(sigma,b1_war_teor_duza)
plt.title('B1 wariancja')
plt.show()
