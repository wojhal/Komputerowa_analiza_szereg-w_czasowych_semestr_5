import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import csv

x = np.loadtxt('/Users/wojtek/Desktop/KASC/1zad2.txt')
y = np.loadtxt('/Users/wojtek/Desktop/KASC/1zad3.txt')

def prosta_sr_ruchoma(X,podstawa):
    T= []
    start,stop,krok = int((podstawa-1)/2),int(len(X)-(podstawa-1)/2),int(len(X)-(podstawa-1))
    ranga = np.linspace(start,stop,krok)
    for t in ranga:
        T.append((1/podstawa)*sum(X[int(t)-1-start:int(t)-1+start]))
    return T
def prosta_regresji(x,y):
    b_1 = np.sum(x*(y-np.mean(y)))/np.sum((x-np.mean(x))**2)
    b_0 = np.mean(y) - b_1 * np.mean(x)
    return b_0, b_1

b0, b1 = prosta_regresji(x, y)
xs = np.linspace(-1,10,1000)
plt.scatter(x, y)
plt.plot(xs, xs * b1 + b0, color="r")
plt.show()

dane2_wygladzone, dane3_wygladzone = prosta_sr_ruchoma(x, 25), prosta_sr_ruchoma(y, 25)
b01, b11 = prosta_regresji(dane2_wygladzone, dane3_wygladzone)
xs2 = np.linspace(-0.5,2.5,1000)
plt.scatter(dane2_wygladzone, dane3_wygladzone)
plt.plot(xs2, b11*xs2 + b01, color="r")
plt.show()