# shapeAI_python_machine_learni
Python_machine_learning project 
In [ ]:
import  nummy as np
import pandas as pd
import sklearn
In [ ]:
from sklearn.dataseets import load_boston
df = load_boston()
In [ ]:
df.keys()
dict_keys(['data','target','feature_names','DESC','filename'])

In [ ]:
boston =  pd.DataFrame(df.data, colums=df.feature_names)
boston.head()
In [ ]:
boston['MEDV'] =  df.target
boston.head()
In [ ]:
boston.isnull()
In [ ]:
boston.isnull().sum()
In [ ]:
from sklearn.model_selection import train_test_split
X=boston.drop('MEDV',axis=1)
Y=boston['MEDV']
X_trin,X_test,Y_tain,Y_test=train_test_split(X,Y,test_size=0.15random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
In [ ]:
from skleaarn.linear_odel import linearegression
from sklarn.metrics import men_squared_error
In [ ]:
lin_model=linearregression()
lin_model.fit(X_train,Y_train)
In [ ]:
y_train_predict=lin_model.predict(X_train)
rmse=(np.sqrt(mean_squared_error(Y_train,y_train_predict)))
print("the model performancr for traing set")
print('RMSE is {}'.format(rmse))
print("\n")
y_test_predict=lin_model.predict(X_test)
rmse=(np.sqrt(mean_squared_error(Y_test,y_test_predict)))
print("the model performance for testing set")
print('RMSE is {}'.format(rmse))
