import numpy as np
import pandas as pd

filename = 'train.csv'
cleaned_filename = 'cleaned_titanic.csv'
dataframe = pd.read_csv(filename)

# extract title from names
titles = {"Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master", "Rare": "Rare"}
dataframe['Title'] = dataframe.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
dataframe['Title'] = dataframe['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataframe['Title'] = dataframe['Title'].replace('Mlle', 'Miss')
dataframe['Title'] = dataframe['Title'].replace('Ms', 'Miss')
dataframe['Title'] = dataframe['Title'].replace('Mme', 'Mrs')
dataframe['Title'] = dataframe['Title'].map(titles)
dataframe['Title'] = dataframe['Title'].fillna(0)

# drop columns
dataframe = dataframe.drop(['PassengerId'], axis=1)
dataframe = dataframe.drop(['Cabin'], axis=1)
dataframe = dataframe.drop(['Ticket'], axis=1)
dataframe = dataframe.drop(['Sex'], axis=1)
dataframe = dataframe.drop(['Name'], axis=1)

# impute age
mean = dataframe["Age"].mean()
std = dataframe["Age"].std()
is_null = dataframe["Age"].isnull().sum()
random_age = np.random.randint(mean - std, mean + std, size = is_null)
age_slice = dataframe["Age"].copy()
age_slice[np.isnan(age_slice)] = random_age
dataframe["Age"] = age_slice
dataframe["Age"] = dataframe["Age"].astype(int)

# standardize age
dataframe['Age'] -= dataframe["Age"].mean()
dataframe['Age'] /= dataframe["Age"].std()

# impute embarked
common_value = 'S'
dataframe['Embarked'] = dataframe['Embarked'].fillna(common_value)

# encode categorical variables
dataframe = pd.concat([pd.get_dummies(dataframe['Title'], prefix='Title', drop_first=True), dataframe], axis=1)
dataframe.drop(['Title'], axis=1, inplace=True)
dataframe = pd.concat([pd.get_dummies(dataframe['Embarked'], prefix='Embarked', drop_first=True), dataframe], axis=1)
dataframe.drop(['Embarked'], axis=1, inplace=True)

# log(x+1) fare
dataframe['Fare'] += 1
dataframe['LogFare'] = np.log(dataframe['Fare'])
dataframe.drop(['Fare'], axis=1, inplace=True)

# move target variable to end
survived = dataframe['Survived']
dataframe.drop(['Survived'], axis=1, inplace=True)
dataframe = pd.concat([dataframe, survived], axis=1)


# export to csv
dataframe.to_csv(cleaned_filename)

print(dataframe.head())
print(dataframe.shape)
print(list(dataframe))