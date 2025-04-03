import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import csv
import scipy.stats as stats
import seaborn as sns
import random
from statsmodels.tsa.arima_process import ArmaProcess
from scipy.optimize import curve_fit

fi1 = 0.2
fi2 = 0.4
sigma = 1

def autokowariancja(h,x):
     return (1/len(x))*np.sum((x[:len(x)-abs(h)]-np.mean(x))*(x[abs(h):]-np.mean(x)))

def jakies_cos_czego_nie_rozumiem(trajektoria,p):
    autokowariancja_p = [autokowariancja(h,trajektoria) for h in range(1,p+1)]
    autokowariancja_p = np.transpose(autokowariancja_p)
    autogamma_p = np.array(np.zeros(p**2)).reshape(p, p)
    for i in range(p):
         for j in range(p):
              autogamma_p[i][j] = autokowariancja(i-j,trajektoria)
    autogamma_p = np.linalg.inv(autogamma_p)
    fi_daszek = np.matmul(autokowariancja_p,autogamma_p)
    sigma = autokowariancja(0,trajektoria) - np.matmul(np.transpose(fi_daszek),autokowariancja_p)
    return fi_daszek,sigma

ps = np.linspace(0,10,11)
ns = [50, 100, 500, 1000]
N = 100
ar_process = ArmaProcess(ar=[1,-fi1,-fi2],ma=1)
wszystkie = []
for nss in ns:
    histogramy = []
    for _ in range(N):
        simulated_data = ar_process.generate_sample(nsample=nss,scale=np.sqrt(sigma))
        rzad = []
        for pss in ps:
             fi1,sigmaa = jakies_cos_czego_nie_rozumiem(simulated_data,int(pss))
             rzad.append((sigmaa*(nss+int(pss)))/(nss-int(pss)))
        rzad_modelu = np.argmin(rzad)
        histogramy.append(rzad_modelu)
    wszystkie.append(histogramy)

plt.hist(wszystkie[0],bins= 20)
plt.show()
plt.hist(wszystkie[1],bins= 20)
plt.show()
plt.hist(wszystkie[2],bins= 20)
plt.show()
plt.hist(wszystkie[3],bins= 20)
plt.show()
