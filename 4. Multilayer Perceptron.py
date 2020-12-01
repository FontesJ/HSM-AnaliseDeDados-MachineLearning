#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 22:46:08 2020

@author: julio
"""

from sklearn.datasets import load_diabetes

pacient_data = load_diabetes()

x= pacient_data['data']
y= pacient_data['target']

#Gerando bases de treino e teste
from sklearn.model_selection import train_test_split

    #Conforme variamos os parâmetros do split, os resultados apresentados são diferentes
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

#Aplicando método Multilayer Perceptron
from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(hidden_layer_sizes=(50, 50), learning_rate_init=0.01, activation="relu", alpha=0.1)
mlp.fit(X_train, y_train)
pred_mlp = mlp.predict(X_test)

#Imprimindo gráfico comparando precisão
from matplotlib import pyplot

plt = pyplot
plt.plot(y_test, color="Red")
plt.plot(pred_mlp, color="blue")
plt.show()

#Exibindo balanceamento
r2_train = mlp.score(X_train, y_train)
r2_test = mlp.score(X_test, y_test)

print('R2 do mlp no set de treino: %.2f' % r2_train)
print('R2 do mlp no set de teste: %.2f' % r2_test)
