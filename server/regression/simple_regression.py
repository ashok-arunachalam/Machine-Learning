#importing the necessary module
import numpy as ns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#importing dataset
dataset = pd.read_csv('D:/Ashok\'s-Poc/Machine-Learning/server/resources/Salary_Data.csv')
feature_value_x = dataset.iloc[ : , :-1].values
label_value_y= dataset.iloc[ : , -1].values

#spliting the data into training and test
feature_train, feature_test, label_train, label_test = train_test_split(feature_value_x, label_value_y, test_size=0.2, random_state=0)

#Train the simple regression model using the training set
regressor = LinearRegression()
regressor.fit(feature_train, label_train)

#predict the label using the feature test values
label_predict = regressor.predict(feature_test)
print(label_predict)

#visualize the training set results
plt.scatter(feature_train, label_train, color = 'red')
plt.plot(feature_train, regressor.predict(feature_train), color = 'blue')
plt.title('Salery vs Experience (Training set)')
plt.xlabel('years')
plt.show()

#visualize the test set results
plt.scatter(feature_test, label_test, color = 'red')
plt.plot(feature_train, regressor.predict(feature_train), color = 'blue')
plt.title('Salery vs Experience (Test set)')
plt.xlabel('years')
plt.show()