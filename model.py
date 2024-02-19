import pandas as pd
import numpy as np
import joblib

df = pd.read_csv("train.csv")
include = ['Age', 'Sex', 'Embarked', 'Survived']
df_ = df[include]
dff = df_.copy()
for col in dff.columns:
    dff.fillna({col: 0}, inplace=True)


categoricals = ['Sex', 'Embarked']
df_ohe = pd.get_dummies(dff, columns=categoricals, dummy_na=True)

from sklearn.linear_model import LogisticRegression
dependent_variable = 'Survived'
x = df_ohe[df_ohe.columns.difference([dependent_variable])]
y = df_ohe[dependent_variable]
lr = LogisticRegression()
lr.fit(x, y)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

joblib.dump(lr, 'model.pkl')
print("Model dumped!")

lr = joblib.load('model.pkl')

model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")
