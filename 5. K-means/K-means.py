#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 00:41:33 2020

@author: julio
"""

import pandas as pd

#Utilizada a mesma base do exercício anterior, porém sem os alvos
base = pd.read_csv("base_dados_clientes.csv")
base = base.drop("rotulo", axis=1)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, max_iter=10)
labels = kmeans.fit_predict(base)
