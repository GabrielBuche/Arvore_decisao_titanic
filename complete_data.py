import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('titanic.csv')
data['Sex'] = data['Sex'].replace({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].replace({'S': 10, 'Q': 20, 'C': 30})

data_completa = data[data['Age'].notna()]
data_completa =  data_completa.dropna()
data_incompleta = data[data['Age'].isna()]

X_complete = data_completa.drop('Age', axis=1)

X_incomplete = data_incompleta.drop('Age', axis=1)
y_complete = data_completa['Age']

arvore_Age = DecisionTreeRegressor()

arvore_Age.fit(X_complete, y_complete)

age_predictions = arvore_Age.predict(X_incomplete)

data_incompleta['Age'] = age_predictions

data_filled = pd.concat([data_completa, data_incompleta])

y_predicted = arvore_Age.predict(X_complete)

mse = mean_squared_error(y_complete, y_predicted)

r2 = r2_score(y_complete, y_predicted)

print("Mean Squared Error (MSE):", mse)
print("R-squared:", r2)



data_filled.to_csv('titanic_tratado.csv', index=False)

