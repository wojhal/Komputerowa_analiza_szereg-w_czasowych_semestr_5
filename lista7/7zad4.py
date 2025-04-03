import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import csv
import scipy.stats as stats
import seaborn as sns
import random
from statsmodels.tsa.arima_process import ArmaProcess, arma_acovf, arma_acf, arma_pacf
from scipy.optimize import curve_fit
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fi1 = 0.2
fi2 = 0.4
sigma = 1
n = 2000

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
ar_process = ArmaProcess(ar=[1,-fi1,-fi2])
simulated_data = ar_process.generate_sample(nsample=n,scale=np.sqrt(sigma))
rzad =[]
for pss in ps:
    fi1,sigmaa = jakies_cos_czego_nie_rozumiem(simulated_data,int(pss))
    rzad.append((sigmaa*(n+int(pss)))/(n-int(pss)))
plt.scatter(ps,rzad)
plt.show()
rzad_modelu = np.argmin(rzad)
fi1_final,sigma_final = jakies_cos_czego_nie_rozumiem(simulated_data,rzad_modelu)

model = ARIMA(simulated_data, order=(rzad_modelu, 0, 1)).fit(method='innovations_mle')
# fitted_values = model.fittedvalues
# plt.plot(simulated_data)
# plt.plot(fitted_values)
# plt.show()
residuals = model.resid

plt.plot(residuals)
plt.show()
plt.hist(residuals,bins=50,density=True,label="residuals_dist")
mu = 0
x = np.linspace(-5,5, 1000)
plt.plot(x, stats.norm.pdf(x, mu, sigma),color="red",label="norm_dist")
plt.legend()
plt.show()
h_max = 20
plot_acf(residuals, lags=h_max)
plt.show()

N=5000
param=[1].extend(fi1_final)
dopasowany_ar_process = ArmaProcess(ar=param)
wszystkie = [[] for c in range(n)]
for _ in range(N):
    nowe_simulated_data = dopasowany_ar_process.generate_sample(nsample=n,scale=np.sqrt(sigma_final))
    # plt.plot(simulated_data)
    # plt.plot(nowe_simulated_data)
    # plt.show()
    for p in range(len(nowe_simulated_data)):
        wszystkie[p].append(nowe_simulated_data[p])
quantils = np.linspace(0.1,0.9,9)
wszystkie_kwantyle = []
for quant in quantils:
    kwantyle = []
    for point in wszystkie:
        kwantyle.append(np.quantile(point,quant))
    wszystkie_kwantyle.append(kwantyle)
quant_plot=np.linspace(1,2000,2000)
plt.plot(simulated_data)
for f in range(9):
    plt.plot(wszystkie_kwantyle[f])
plt.show()
licznik_0=0
licznik_1=0
licznik_2=0
licznik_3=0
licznik_4=0
licznik_5=0
licznik_6=0
licznik_7=0
licznik_8=0
licznik_9=0
for ns in range(n):
    probkowy_punkt = simulated_data[ns]
    if probkowy_punkt <= wszystkie_kwantyle[0][ns]:
        licznik_0 += 1
    elif wszystkie_kwantyle[0][ns] < probkowy_punkt <= wszystkie_kwantyle[1][ns]:
        licznik_1 += 1
    elif wszystkie_kwantyle[1][ns] < probkowy_punkt <= wszystkie_kwantyle[2][ns]:
        licznik_2 += 1
    elif wszystkie_kwantyle[2][ns] < probkowy_punkt <= wszystkie_kwantyle[3][ns]:
        licznik_3 += 1
    elif wszystkie_kwantyle[3][ns] < probkowy_punkt <= wszystkie_kwantyle[4][ns]:
        licznik_4 += 1
    elif wszystkie_kwantyle[4][ns] < probkowy_punkt <= wszystkie_kwantyle[5][ns]:
        licznik_5 += 1
    elif wszystkie_kwantyle[5][ns] < probkowy_punkt <= wszystkie_kwantyle[6][ns]:
        licznik_6 += 1
    elif wszystkie_kwantyle[6][ns] < probkowy_punkt <= wszystkie_kwantyle[7][ns]:
        licznik_7 += 1
    elif wszystkie_kwantyle[7][ns] < probkowy_punkt <= wszystkie_kwantyle[8][ns]:
        licznik_8 += 1
    elif wszystkie_kwantyle[8][ns] < probkowy_punkt:
        licznik_9 += 1
liczniki=[100*licznik_0/n,100*licznik_1/n,100*licznik_2/n,100*licznik_3/n,100*licznik_4/n,100*licznik_5/n,100*licznik_6/n,100*licznik_7/n,100*licznik_8/n,100*licznik_9/n]
quant_s = np.linspace(10,100,10)
plt.plot(quant_s,liczniki)
plt.show()


    