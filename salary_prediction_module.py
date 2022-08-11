import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

salary_data=pd.read_csv('salary_Data.csv')

x=salary_data.iloc[:,0:1].values
y=salary_data.iloc[:,1:2].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test,=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
salary_model=LinearRegression()
salary_model.fit(x_train,y_train)

salary_prediction =salary_model.predict(x_test)
