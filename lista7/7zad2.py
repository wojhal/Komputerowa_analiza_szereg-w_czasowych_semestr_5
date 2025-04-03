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
M = 100
fi = 0.1
sigma = 0.5
ns = np.linspace(50,1000,20)
def wartosci(trajektoria):
    suma_fi = 0
    for i in range(1,len(trajektoria)):
          suma_fi += trajektoria[i]*trajektoria[i-1]
    fi = suma_fi/np.sum(trajektoria**2)
    suma_sigm = 0
    for j in range(1,len(trajektoria)):
         suma_sigm += (trajektoria[j]-fi*trajektoria[j-1])**2
    sigma = suma_sigm/len(trajektoria)
    return fi,sigma
ar_process = ArmaProcess(ar=[1,-fi])
sigmas = []
fis = []
for nss in ns:
    fi_yws = []
    fi_nws = []
    sigma_yws = []
    sigma_nws = []
    for _ in range(M):
        simulated_data = ar_process.generate_sample(nsample=int(nss),scale=np.sqrt(0.5))
        fi_nw,sigma_nw = wartosci(simulated_data)
        fi_yw, sigma_yw = jakies_cos_czego_nie_rozumiem(simulated_data,1)
        fi_yws.append(fi_yw[0])
        fi_nws.append(fi_nw)
        sigma_yws.append(sigma_yw)
        sigma_nws.append(sigma_nw)
    sigmas.append(sigma_yws)
    sigmas.append(sigma_nws)
    sigmas.append(np.zeros(int(nss)))
    sigmas.append(np.zeros(int(nss)))
    fis.append(fi_yws)
    fis.append(fi_nws)
    fis.append(np.zeros(int(nss)))
    fis.append(np.zeros(int(nss)))

plt.boxplot(fis)
plt.plot()
plt.show()

plt.boxplot(sigmas)
plt.plot()
plt.show()