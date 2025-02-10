# Copyright 2025 Ashok

#imppoting the libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


#importing dataSets
datasets = pd.read_csv('D:/Ashok\'s-Poc/Machine-Learning/server/resources/Data.csv')
x = datasets.iloc[ :, :-1].values
y = datasets.iloc[: , -1].values

#print(x!=NULL)

# Taking care of missing values and replaces by the mean of the existing values in the coloum
# using the inbuilt functions in python.

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[ : , 1:3])
x[:, 1:3] = imputer.transform(x[ : , 1:3])

print(x)