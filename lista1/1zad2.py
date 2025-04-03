import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

c = np.loadtxt('/Users/wojtek/Desktop/KASC/1zad2.txt')
def prosta_sr_ruchoma(X,podstawa):
    T= []
    start,stop,krok = int((podstawa-1)/2),int(len(X)-(podstawa-1)/2),int(len(X)-(podstawa-1))
    ranga = np.linspace(start,stop,krok)
    print(ranga)
    for t in ranga:
        T.append((1/podstawa)*sum(X[int(t)-1-start:int(t)-1+start]))
    return T

ma11 = prosta_sr_ruchoma(c,11)
ma25 = prosta_sr_ruchoma(c,25)
ma4001 = prosta_sr_ruchoma(c,4001)
plt.plot(np.linspace(1,10000,10000),c)
plt.plot(np.linspace(5,10000-5,10000-10),ma11,label='11')
plt.plot(np.linspace(12,10000-12,10000-24),ma25,label='25')
plt.plot(np.linspace(2000,10000-2000,10000-4000),ma4001,label='4001')
plt.legend()
plt.show()
