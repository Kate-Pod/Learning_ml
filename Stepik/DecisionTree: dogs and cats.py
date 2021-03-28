#Скачайте тренировочный датасэт и  обучите на нём Decision Tree. 
#После этого скачайте датасэт из задания и предскажите какие наблюдения к кому относятся. 
#Введите число собачек в вашем датасэте.
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split

train_data=pd.read_csv('C:/Users/.../dogs_n_cats.csv')
train_data.head()
y=train_data.Вид
X=train_data.drop(['Вид'],axis=1)
clf=tree.DecisionTreeClassifier(criterion='entropy')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
X_train.shape #670 наблюдений
X_test.shape #330 наблюдений
clf.score(X,y) #1
clf.fit(X_train,y_train) 
clf.score(X_train,y_train) #1
clf.score(X_test, y_test) #1

df_ts=pd.read_json(r'C:/Users/.../dataset_209691_15.txt')
result=clf.predict(df_ts)
pd.Series(result)[result=='собачка'].count()
