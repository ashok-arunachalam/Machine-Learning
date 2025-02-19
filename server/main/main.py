# Copyright 2025 Ashok

#impoting the libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
#for one hot encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
#imports for data splitting train and test
from sklearn.model_selection import train_test_split

#<------------------------------------------------------------------------------------------------------------------------------->
#importing dataSets
datasets = pd.read_csv('D:/Ashok\'s-Poc/Machine-Learning/server/resources/Data.csv')
feature_value_x = datasets.iloc[ :, :-1].values
label_value_y = datasets.iloc[: , -1].values

#<------------------------------------------------------------------------------------------------------------------------------->
# Taking care of missing values and replaces by the mean of the existing values in the coloum
# using the inbuilt functions in python.

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(feature_value_x[ : , 1:3])
feature_value_x[:, 1:3] = imputer.transform(feature_value_x[ : , 1:3])
print('Printing value after updating the missing value: \n', feature_value_x)

#<------------------------------------------------------------------------------------------------------------------------------->
#Encode the categorical variables into a binary format!!!
#One Hot Encoding is a method for converting categorical variables into a binary format.
# It creates new columns for each category where 1 means the category is present and 0 means it is not.
# The primary purpose of One Hot Encoding is to ensure that categorical data can be effectively used in machine learning models.
#https://www.geeksforgeeks.org/ml-one-hot-encoding/


# Encoding the indepent variable like country (feature value)
#Method -1 mentioning the coloum name by using the entire datasets
coloum_tranformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['Country'])], remainder='passthrough')
encoded_feature_value_x = coloum_tranformer.fit_transform(datasets)
print("Method - 1: Printing the table after converting the categorical variables into a binary format: \n",encoded_feature_value_x)

# Method -2 mentioning the coloum index by mentioning the extracted datasets from the datasets like feature_value_x
coloum_tranformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
encoded_feature_value_x = coloum_tranformer.fit_transform(feature_value_x)
print("Method -2: Printing the table after converting the categorical variables into a binary format: \n",encoded_feature_value_x)

# Encoding the dependent variable like purchased (label value)
label_transformer = LabelEncoder()
encoded_label_value_y = label_transformer.fit_transform(label_value_y)
print(encoded_label_value_y)

#<-------------------------------------------------------------------------------------------------------------------------------->
feature_value_x_train, feature_value_x_test, label_value_y_train, label_value_y_test = train_test_split(feature_value_x, label_value_y,
                                                                                                        test_size= 0.2, random_state=1)

print(feature_value_x_train, feature_value_x_test, label_value_y_train, label_value_y_test)
