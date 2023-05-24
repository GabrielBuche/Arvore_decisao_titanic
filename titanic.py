import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pickle
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('titanic_tratado.csv')
data['Sex'] = data['Sex'].replace({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].replace({'S': 10, 'Q':20, 'C': 30})

X = data.drop('Survived', axis=1)
Y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

arvoreDecisao = DecisionTreeClassifier(criterion='entropy',max_depth=6)

arvoreDecisao.fit(X_train, y_train)

accuracy = arvoreDecisao.score(X_test, y_test)

print(accuracy * 100)

with open("arvore_decisao.pkl", "wb") as f:
    pickle.dump(arvoreDecisao, f)

tree.plot_tree(arvoreDecisao)
plt.show()