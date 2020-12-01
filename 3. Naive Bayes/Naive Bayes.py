#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 22:43:40 2020

@author: julio
"""

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

base = pd.read_csv("base_dados_clientes.csv")

x = base.iloc[:, 0:len(base.columns)-1]
y = base['rotulo']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

classificadorNaiveBayes = GaussianNB()
classificadorNaiveBayes.fit(x_train, y_train)
y_pred =  classificadorNaiveBayes.predict(x_test)

print(classification_report(y_test, y_pred, target_names=['mau pagador', 'bom pagador']))
