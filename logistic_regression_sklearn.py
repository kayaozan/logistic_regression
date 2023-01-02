# This is a sample code of logistic regression performed via scikit-learn, written by Ozan KAYA.
# https://github.com/kayaozan

# The dataset is called Titanic.
# It contains several data of passengers such as sex, age, fare, and whether they survived or not.
# The dataset can be obtained from https://www.openml.org/search?type=data&sort=runs&id=40945

# The aim of this code is to estimate the parameters of logistic model
# and predict the survival outcome.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('titanic3.csv',index_col=False)

# Replacing text variables with binary values
# Replacing any unknown data with NaN
data.replace({'male': 1, 'female': 0}, inplace=True)
data.replace('?', np.nan, inplace= True)

# Dropping the rows that contain NaN and selecting the columns that will be used
data = data[['sex', 'pclass','age','fare','survived']].dropna()

# Splitting the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    (data.loc[:,data.columns != 'survived']),
     data.survived,
     test_size=0.2,
     random_state=0
     )

# Normalizing the data before processing
sc = StandardScaler()
x_train = sc.fit_transform(x_train)

# Normalizing the test set after fitting with training set
# so that the model is not biased
x_test = sc.transform(x_test)

# Training the model
logReg = LogisticRegression()
logReg.fit(x_train, y_train)

# Calculating and printing the accuracy
print('The accuracy measured with test set is:\n', logReg.score(x_test, y_test))