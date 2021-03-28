import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import tree

from graphviz  import Source

titanic_data=pd.read_csv('C:/Users/.../Downloads/train.csv')
titanic_data.isnull().sum() #проверим кол-во пропущенных значений по переменным
X=titanic_data.drop(['PassengerId','Survived','Name','Ticket','Cabin'], axis=1) #удалим ненужные для вычисления закономерностей колонки
y=titanic_data.Survived #переменная, которую будем предсказывать
X=pd.get_dummies(X)
X=X.fillna({'Age': X.Age.median()}) #заменим пропущенные значения медианой возраста

clf=tree.DecisionTreeClassifier(criterion='entropy') #работает только с числовыми переменными, поэтому номинативные надо закодировать как 0 и 1, например
clf.fit(X,y)

#Нарисуем дерево
import os
os.environ["PATH"] += os.pathsep + 'E:/Anaconda3/Library/bin/graphviz'
graph=Source(tree.export_graphviz(clf,  feature_names=list(X),out_file="tree.dot", 
                                  class_names=['Died', 'Survived'], filled=True))

with open("tree.dot") as f:
    dot_graph = f.read()
g=Source(dot_graph)
g.format = "png"
g.render("file_name") #дерево получилось слишком глубоким и большим



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42) #тестовое множество 33%
X_train.shape #596 наблюдений
X_test.shape #295 наблюдений
#Посмотрим, насколько плохо было получившееся дерево
clf.score(X,y) #0.9797979797979798
clf.fit(X_train,y_train) #обучим часть модели и посмотрим точность классификации на обучающем множестве
clf.score(X_train,y_train) #0.9798657718120806
#Какая точность на данных, который классификатор не "видел"?:
clf.score(X_test, y_test) #0.7728813559322034
'''Возможно объяснение данной оценки кроется в том, что мы слишком переобучили дерево: вместо расчета закономерности из данных?
которую можно обобщить и применить на новые данные, дерево пыталось как можно лучше решить конкретную задачу - мы не 
ограничили количество ресурсов, которое дерево может потратить: макс глубину.'''
clf=tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
clf.fit(X_train,y_train)
clf.score(X_train,y_train) #0.8406040268456376 - на train классификатор стал работать хуже
clf.score(X_test, y_test) #0.8067796610169492 - для тестовой выборки классификатор стал работать лучше

