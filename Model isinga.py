#!/usr/bin/env python
from numpy.random import *
import  numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import statsmodels.api as sm
import pylab as py
import seaborn as sns
import random
import math
from scipy.sparse import spdiags,linalg,eye
import numba
from numba import jit,cuda,float32
import random
import time
import json


"""Generuje losową siatke 1 i -1"""
def generuj_siatke(L):
    siatka = 2*np.random.randint(2,size=(L,L))-1
    return siatka

"""Jeden krok monte carlo dla modelu Isinga"""
@jit(nopython=True)
def krok_isinga(siatka, T, J, kB):
    if T==0:
        return siatka
    L = len(siatka[0])
    for k in range(L):
        for p in range(L):
            i = np.random.randint(0, L)
            j = np.random.randint(0, L)
            delta_E = 2*J*siatka[i][j] * ( siatka[(i-1)%L, j] + siatka[(i+1)%L, j] + siatka[i, (j-1)%L] + siatka[i, (j+1)%L])
            if delta_E <= 0:
                siatka[i][j] = -siatka[i][j]
            else:
                x = rand()
                if x < np.exp(-delta_E/(kB*T)):
                    siatka[i][j] = -siatka[i][j]
    return siatka

"""Magnetyzacja"""
def magnetyzacja1(siatka):
    mag = np.mean(siatka)
    return mag

"""Magnetyzacja danych z listy"""
def magnetyzacja2(list,K0):
    mag = np.mean(np.abs(list[K0:]))
    return mag

"""Podatność danych z listy"""
def podatnosc(list, K0, T, N):
    m1 = magnetyzacja2(np.square(list), K0)
    m2 = magnetyzacja2(list, K0)
    pod = N/T*(m1-m2**2)
    return pod



"""Konfiguracje spinów dla L=10,100 dla różnych T do zadania 1"""
def spiny_do_1(K, L):
    J = 1
    kB = 1 
    T1 = [1]
    T2 = [2.26]
    T3 = [4]

    for t in T3:
        siatka = generuj_siatke(L)
        X, Y = np.meshgrid(range(L+1), range(L+1))
        plt.pcolormesh(X, Y, siatka, cmap='tab10')
        plt.show()
        for i in range(K):
            krok_isinga(siatka, t, J, kB)
            if i==100:
                X, Y = np.meshgrid(range(L+1), range(L+1))
                plt.pcolormesh(X, Y, siatka, cmap='tab10')
                plt.show()
            if i==1000:
                X, Y = np.meshgrid(range(L+1), range(L+1))
                plt.pcolormesh(X, Y, siatka, cmap='tab10')
                plt.show()
            if i==5000:
                X, Y = np.meshgrid(range(L+1), range(L+1))
                plt.pcolormesh(X, Y, siatka, cmap='tab10')
                plt.show()
            if i==10000:
                X, Y = np.meshgrid(range(L+1), range(L+1))
                plt.pcolormesh(X, Y, siatka, cmap='tab10')
                plt.show()
            if i==50000:
                X, Y = np.meshgrid(range(L+1), range(L+1))
                plt.pcolormesh(X, Y, siatka, cmap='tab10')
                plt.show()
#spiny_do_1(50000,100)



"""Generowanie i wyświetlanie trejektorii dla T=1.7 do zadania 2"""
def trajektorie_do_2(K):
    T = 1.7
    J = 1
    kB = 1

    siatka = generuj_siatke(10)
    trajektoria10 = [magnetyzacja1(siatka)]
    for i in range(K):
        krok_isinga(siatka, T, J, kB)
        trajektoria10.append(magnetyzacja1(siatka))

    siatka = generuj_siatke(50)
    trajektoria50 = [magnetyzacja1(siatka)]
    for i in range(K):
        krok_isinga(siatka, T, J, kB)
        trajektoria50.append(magnetyzacja1(siatka))

    siatka = generuj_siatke(100)
    trajektoria100 = [magnetyzacja1(siatka)]
    for i in range(K):
        krok_isinga(siatka, T, J, kB)
        trajektoria100.append(magnetyzacja1(siatka))

    file = open("Trajektoria10.json", "w")
    json.dump(trajektoria10, file)
    file.close()

    file = open("Trajektoria50.json", "w")
    json.dump(trajektoria50, file)
    file.close()

    file = open("Trajektoria100.json", "w")
    json.dump(trajektoria100, file)
    file.close()


    plt.plot(trajektoria10, color="purple", label = 'L=10')
    plt.xlabel("t[MCS]")
    plt.ylabel("m")
    plt.title("Trajektoria dla T=1.7")
    plt.legend()
    plt.show()

    plt.plot(trajektoria50, color="purple", label = 'L=50')
    plt.xlabel("t[MCS]")
    plt.ylabel("m")
    plt.title("Trajektoria dla T=1.7")
    plt.legend()
    plt.show()

    plt.plot(trajektoria100, color="purple", label = 'L=100')
    plt.xlabel("t[MCS]")
    plt.ylabel("m")
    plt.title("Trajektoria dla T=1.7")
    plt.legend()
    plt.show()
#trajektorie_do_2(10000)


"""Symulacja monte carlo dla wyznaczenia magnetyzacji i wczytywanie danych do pliku tekstowego"""
def monte_carlo_do_3(K, L):
    #data=open("L100.json","r")
    #wyniki=json.loads(data.read())
    wyniki = {}
    kB = 1
    J = 1
    a = np.linspace(1, 2, 5)
    b = np.linspace(2.1, 3, 15)
    c = np.linspace(3.1, 3.5, 5)
    T = np.hstack([a,b,c])
    for t in T:
        siatka = generuj_siatke(L)
        lista = [magnetyzacja1(siatka)]
        for i in range(K):
            krok_isinga(siatka, t, J, kB)
            lista.append(magnetyzacja1(siatka))
        wyniki[str(t)] = lista
    file = open("L10.json", "w")
    json.dump(wyniki, file)
    file.close()
#monte_carlo_do_3(1000000, 10)


"""Wykresy magnetyzacji do zadania 3"""
def wykresy_do_3(plik1, plik2, plik3):
    a = np.linspace(1, 2, 5)
    b = np.linspace(2.1, 3, 15)
    c = np.linspace(3.1, 3.5, 5)
    T=np.hstack([a,b,c])

    data = open(plik1,"r")
    dane10 = json.loads(data.read())
    mag10 = []
    for i in T:
        d=dane10[str(i)]
        mag10.append(magnetyzacja2(d,10000))
    data.close()

    data = open(plik2,"r")
    dane50 = json.loads(data.read())
    mag50 = []
    for i in T:
        d=dane50[str(i)]
        mag50.append(magnetyzacja2(d,10000))
    data.close()

    data = open(plik3,"r")
    dane100 = json.loads(data.read())
    mag100 = []
    for i in T:
        d=dane100[str(i)]
        mag100.append(magnetyzacja2(d,10000))
    data.close()

    plt.plot(T, mag10,marker="o", color='blue', label = 'L=10')
    plt.plot(T, mag50, marker="o", color='orange', label = 'L=50')
    plt.plot(T, mag100, marker="o", color='gray', label = 'L=100')
    plt.xlabel("T* - zredukowana temperatura")
    plt.ylabel("<m> - magnetyzacja")
    plt.title("<m>(T*) - Magnetyzacja od temperatury")
    plt.legend()
    plt.grid(True)
    plt.show()
#wykresy_do_3("L10.json","L50.json","L100.json")


"""Wykresy podatności magnetycznej do zadania 4"""
def wykresy_do_4(plik1, plik2, plik3):
    a = np.linspace(1, 2, 5)
    b = np.linspace(2.1, 3, 15)
    c = np.linspace(3.1, 3.5, 5)
    T=np.hstack([a,b,c])

    data = open(plik1,"r")
    dane10 = json.loads(data.read())
    pod10 = []
    for i in T:
        d=dane10[str(i)]
        pod10.append(podatnosc(d,10000,i,100))
    data.close()

    data = open(plik2,"r")
    dane50 = json.loads(data.read())
    pod50 = []
    for i in T:
        d=dane50[str(i)]
        pod50.append(podatnosc(d,10000,i,2500))
    data.close()

    data = open(plik3,"r")
    dane100 = json.loads(data.read())
    pod100 = []
    for i in T:
        d=dane100[str(i)]
        pod100.append(podatnosc(d,10000,i,10000))
    data.close()

    plt.plot(T, pod10,marker="o", color='blue', label = 'L=10')
    plt.plot(T, pod50, marker="o", color='orange', label = 'L=50')
    plt.plot(T, pod100, marker="o", color='gray', label = 'L=100')
    plt.xlabel("T* - zredukowana temperatura")
    plt.ylabel("X - podatność")
    plt.title(" X(T*) - Podatność magnetyczna od temperatury")
    plt.legend()
    plt.grid(True)
    plt.show()
wykresy_do_4("L10.json","L50.json","L100.json")







   


                


    
