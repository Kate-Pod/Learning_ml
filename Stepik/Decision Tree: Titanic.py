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


max_depth_values=range(1, 100)
scores_data=pd.DataFrame() #создадим пустой дф для сохранения в него результатов обучения
for max_depth in max_depth_values: #для каждого значения глубины дерева в массиве выше
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth) #инициировать класс-ор с указанной глубиной
    clf.fit(X_train, y_train) #обучаться на трейн выборке
    train_score = clf.score(X_train, y_train) #предсказывать точность классификации на этой же выборке
    test_score = clf.score(X_test, y_test) #и на тестовой
    
    temp_score_data = pd.DataFrame({'max_depth': [max_depth], 'train_score': [train_score], 'test_score': [test_score]})

    scores_data=scores_data.append(temp_score_data)
 
scores_data.head()
scores_data_long=pd.melt(scores_data, id_vars=['max_depth'], value_vars=['train_score','test_score'],
                                              var_name='set_type', value_name='score')
scores_data_long.head() #теперь для каждой глубины дерева по два значения: для set_type: train and test

sns.lineplot(x='max_depth', y='score',hue='set_type',data=scores_data_long) #посмотрим зависимость точности от глубины
plt.show()
'''С увеличением количества деревьев данные обучаются точнее -train_score. Но с увеличеним train_score и глубины test_score
постепенно снижается. Где-то на промежутке 0-2 дерево недообучено, далее оптимальное состояние, а дальнейшее углубление дерева
приводит к переобучению модели'''

#КРОСС-ВАЛИДАЦИЯ
from sklearn.model_selection import cross_val_score
clf=tree.DecisionTreeClassifier(criterion='entropy',max_depth=4)
cross_val_score(clf, X_train, y_train, cv=5) 
'''array([0.76666667, 0.82352941, 0.78991597, 0.75630252, 0.80672269]) - это точность классификатора при разбиении X_train & y_train
на 5 равных частей'''
cross_val_score(clf, X_train, y_train, cv=5).mean() #средняя точность

#Повторим вычисление точностей
max_depth_values=range(1, 100)
scores_data=pd.DataFrame()
for max_depth in max_depth_values: 
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth) 
    clf.fit(X_train, y_train) 
    train_score = clf.score(X_train, y_train) 
    test_score = clf.score(X_test, y_test)

    mean_cross_val_score=cross_val_score(clf, X_train, y_train, cv=5).mean()
    
    temp_score_data = pd.DataFrame({'max_depth': [max_depth], 'train_score': [train_score], 'test_score': [test_score],
                                    'cross_val_score':[mean_cross_val_score]})

    scores_data=scores_data.append(temp_score_data)
scores_data.head()
scores_data_long=pd.melt(scores_data, id_vars=['max_depth'], value_vars=['train_score','test_score', 'cross_val_score'],
                                              var_name='set_type', value_name='score')
scores_data_long.head()

sns.lineplot(x='max_depth', y='score',hue='set_type',data=scores_data_long) #посмотрим зависимость точности от глубины
plt.show()
scores_data_long.query("set_type=='cross_val_score'").head(20) #видим при какой глубине макс значение кросс-валидации
best_clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=6)
best_clf.fit(X_train, y_train)
best_clf.score(X_test, y_test) #0.7661016949152543


#Подбор параметров
from sklearn.model_selection import GridSearchCV
parameters={'criterion': ['gini', 'entropy'], 'max_depth':range(1,30)}
grid_search_cv_clf=GridSearchCV(clf,parameters, cv=5)
grid_search_cv_clf.fit(X_train, y_train)
grid_search_cv_clf.best_params_
best_clf=grid_search_cv_clf.best_estimator_
best_clf.score(X_test, y_test) #0.8067796610169492

from sklearn.metrics import precision_score, recall_score
y_pred=best_clf.predict(X_test)
precision_score(y_test, y_pred) #0.8387096774193549
recall_score(y_test, y_pred) #0.65
