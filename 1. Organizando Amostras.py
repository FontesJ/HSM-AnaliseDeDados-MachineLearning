#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 23:01:12 2020

@author: julio
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pandas as pd

iris = load_iris()

base = pd.DataFrame(data=iris.get('data'), columns=[iris.get('feature_names')]) 
base['target'] = iris.get('target')

######     HOLDOUT
base_treino_h, base_teste_h = train_test_split(base, test_size=0.33)

######      CROSS-VALIDATION
kf = KFold(n_splits=2)
cross_v = kf.split(base)

base_treino_cv = []
base_teste_cv = []

for train, test in cross_v:
    base_treino_cv.append(train)
    base_teste_cv.append(test)
    
