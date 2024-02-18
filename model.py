import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib
import pickle

# Load the dataset
df = pd.read_csv("train.csv")

# Select features and target
include = ['Age', 'Sex', 'Embarked', 'Survived']
df_ = df[include]

# Create a copy of the DataFrame
dff = df_.copy()

# Handle missing values
dff['Age'] = dff['Age'].fillna(dff['Age'].median())
dff['Embarked'] = dff['Embarked'].fillna(dff['Embarked'].mode()[0])


# Define categorical columns
categoricals = ['Sex', 'Embarked']

# Create a pipeline for preprocessing
numeric_features = ['Age']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))])

categorical_features = ['Sex', 'Embarked']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Append classifier to preprocessing pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression())])

# Define X and y
X = df_.drop('Survived', axis=1)
y = df_['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
clf.fit(X_train, y_train)

# Save the model
joblib.dump(clf, 'model.pkl')
print("Model dumped!")
