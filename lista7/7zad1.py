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
fi1= 0.2
fi2 = 0.4
sigma = 1
ar_process = ArmaProcess(ar=[1,-fi1,-fi2],ma=1)

N = 1000
fi11 = []
fi21 = []
sigmas = []
for _ in range(N):
     simulated_data = ar_process.generate_sample(nsample=1000)
     fi_daszek,sigma=jakies_cos_czego_nie_rozumiem(simulated_data,2)
     fi11.append(fi_daszek[0])
     fi21.append(fi_daszek[1])
     sigmas.append(sigma)
# plt.axhline(y=0.2, color='r', linestyle='-')
plt.boxplot(fi11)
plt.show()
# plt.axhline(y=0.4, color='r', linestyle='-')
plt.boxplot(fi21)
plt.show()
# plt.axhline(y=1, color='r', linestyle='-')
plt.boxplot(sigmas)
plt.show()