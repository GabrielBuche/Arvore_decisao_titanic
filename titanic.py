import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

data = pd.read_csv('titanic.csv')

def tratar_dados(data):

    # Troca sexo masculino para 0 e feminino para 1
    # Troca a  cidade Cherbourg (C) para 0
    # Troca a  cidade Queenstown (Q) para 1
    # Troca a  cidade Southampton (S) para 2

    data['Sex'] = data['Sex'].replace({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].replace({'C': 0, 'Q': 1, 'S': 2})
    data = data.drop(['PassengerId','Name','Ticket','Cabin'])

    data_completa = data[data['Age'].notna()]
    data_completa =  data_completa.dropna()
    data_incompleta = data[data['Age'].isna()]

    X_complete = data_completa.drop('Age', axis=1)

    X_incomplete = data_incompleta.drop('Age', axis=1)
    y_complete = data_completa['Age']

    arvore_Age = DecisionTreeRegressor(criterion='entropy',max_depth=6)

    arvore_Age.fit(X_complete, y_complete)

    age_predictions = arvore_Age.predict(X_incomplete)

    data_incompleta['Age'] = age_predictions

    data_filled = pd.concat([data_completa, data_incompleta])

    return data_filled

def gerar_arvore_decisao():
    data = pd.read_csv('titanic_tratado.csv')

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

def gerar_perceptron():
    print('perceptron')