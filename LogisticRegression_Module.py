#importing numpy as np
import numpy as np

import pandas as pd

#importing matplotlib.pyplot as plt
import matplotlib.pyplot as plt

#this code is to create a variable to store dataset
dataset = pd.read_csv("Social_Network_Ads.csv")


#create variable x to store the independent values
x = dataset.iloc[:,0:4].values
y = dataset.iloc[:,4:5].values

#Encoding categorical data 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
column_trans = make_column_transformer((OneHotEncoder(), [1]),remainder='passthrough')
x = column_trans.fit_transform(x)
                                        
#spliting the Dataset into Train data and Test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)       

#scaling using StandardScaler 
from sklearn.preprocessing import StandardScaler 

#assigning the StndardScaler to a variable sc_x
sc_x = StandardScaler()

#fitting and transforming the x train and x test
x_train = sc_x.fit_transform(x_train)  


#fittin x_test
x_test = sc_x.transform(x_test)                               
#training the  LogisticRegression module
from sklearn.linear_model import LogisticRegression

#creating a variable and assigning
SocialNetwork_Module=LogisticRegression(random_state=1)

SocialNetwork_Module.fit(x_train,y_train)

#making a prediction
prediction_result = SocialNetwork_Module.predict(x_test)
prediction_result

#Evaluating the answers by accuracy_score
from sklearn.metrics import accuracy_score

Nicholas=confusion_matrix(y_test,prediction_result)

Nicholas

from sklearn.metrics import accuracy_score

#apply commit
score= accuracy_score(y_test,prediction_result)
score
print(score*100,'%')

