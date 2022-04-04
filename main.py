# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 09:07:26 2021
@author: Nguyen Minh Duc
"""
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import requests
# import tensorflow as tf
from datetime import datetime, date
from  datetime import timedelta

symb = input("Saisir la cryptomonnaie (BTC, ETH, ADA): ").upper()
lim = int(input("Saisir le nombre de derniers jours (100, 150, 200 ...500...) : "))
num = int(input("Saisir le nombre de jours de prédiction (5, 10, 15 ..., 30...) : "))
# charger & traiter les données d'origine
endpoint = 'https://min-api.cryptocompare.com/data/histoday?fsym='+symb+'&tsym=EUR&limit='+str(lim)
res = requests.get(endpoint)
x = res.json()
# print(x)

#charger & traiter les données d'origine
# df = pd.read_csv('apple_1a.csv')
# Pour charger le tableau ou String en format JSON
# df = json.loads(x)
df = pd.json_normalize(x["Data"])
#print(x["Data"])
#print(df)

# qq traitements pour faciliter le travail
# df = df.replace({'\$':''}, regex = True)
df = df.astype({"high": float})
df = df.astype({"low": float})
df = df.astype({"close": float})
df = df.astype({"open": float})
df["time"] = pd.to_datetime(df.time, unit='s')
#df=df.set_index("time", inplace=True)
df=df.set_index("time")

NbLines=df.shape[0]
PrixOrg = df["close"].iloc[(NbLines-num):]
##Code1 : Pour montrer qq traitements de DataFrame
df=df.drop(["volumefrom"],axis=1) # supprimer la colonne Volume, car non nécessaire
df=df.drop(["volumeto"],axis=1) # supprimer la colonne Volume, car non nécessaire
df=df.drop(["conversionType"],axis=1) # supprimer la colonne Volume, car non nécessaire
df=df.drop(["conversionSymbol"],axis=1) # supprimer la colonne Volume, car non nécessaire

df["prediction"]=df["close"].shift(-num)    # ajouter une colonne label pour stocker la prédict : décaler avec Close
#print(df.head)
#Data = df.drop()
Data = df.filter(['close', 'open', 'high','low'], axis=1)
#Data = df.drop(["prediction"],axis=1)    #on crée les données dans dataframe Data (Close/Last, Open, High, Low)
                                    #!!!on garder df
#print(df)
X=Data.values #récupère les valeurs
#print(X)
X=preprocessing.scale(X)    #scale : normaliser les données
#print(X)
X=X[:-num]
df.dropna(inplace=True)
Target = df.prediction
Y=Target.values      #récupère les valeurs de colonne Label et les mettre en Y
#print(Y)
#print(np.shape(X), np.shape(Y))  # les dimensions de données
#modèle LinearRegression
#créer les données de train, de test...
#(maths/stat, etc. ???)
#valeur du Bitcoin
facteur = 0.92
if symb =='ADA' :
    facteur = 0.7
elif symb =='ETH' :
    facteur = 0.8

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=facteur)
lr=LinearRegression()
lr.fit(X_train,Y_train)
lr.score(X_test,Y_test)
#print("R^2",X_test,Y_test)
#prédiction
X_predict=X[-num:]
Forecast=lr.predict(X_predict)
#graphe : comparer donnees Org (red) et Prediction (blue)
Ypredict=np.array(Forecast)

X=np.arange(num) # axis X
plt.plot(X,Ypredict)
YOrg=np.array(PrixOrg)
plt.plot(X,YOrg,color='red')

#Creer la liste des dates à afficher
dateX = np.array([])
for i in range (0, num) :
    dateStr = date.today() + timedelta(days=i+1)
    dateX = np.append(dateX, dateStr.strftime("%d/%m/%Y"))

#Remplacer les chiffres sur l'axe des abscisses par des dates avec faire une rotation du libellé (la date) de 45°
plt.xticks(X, dateX, rotation=45)
#affiche la figure a l'ecran
plt.show()