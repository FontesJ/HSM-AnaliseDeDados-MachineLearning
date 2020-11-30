#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 22:38:05 2020

@author: julio
"""

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

# Carregando uma base de dados desbalanceada disponível na biblioteca Scikit-learn

caracteristicas, classe = make_classification(n_classes=2, class_sep=4, n_samples=1000, weights=[0.1, 0.9])

print('Tamanho original da base de dados: %s' % Counter(classe))

sm = SMOTE()
caracteristicas_balanceadas, classes_balanceadas = sm.fit_resample(caracteristicas, classe)

print('Tamanho da base de dados após o balanceamento: %s' % Counter(classes_balanceadas))
