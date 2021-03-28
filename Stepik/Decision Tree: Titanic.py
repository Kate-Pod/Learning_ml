import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import tree

from graphviz  import Source
titanic_data=pd.read_csv('C:/Users/Администратор.WIN-U1NLG8MM702/Downloads/train.csv')
titanic_data.isnull().sum() #проверим кол-во пропущенных значений по переменным
X=titanic_data.drop(['PassengerId','Survived','Name','Ticket','Cabin'], axis=1) #удалим ненужные для вычисления закономерностей колонки
y=titanic_data.Survived #переменная, которую будем предсказывать
X=pd.get_dummies(X)
X=X.fillna({'Age': X.Age.median()}) #заменим пропущенные значения медианой возраста

clf=tree.DecisionTreeClassifier(criterion='entropy') #работает только с числовыми переменными, поэтому номинативные надо закодировать как 0 и 1, например
clf.fit(X,y)
graph=Source(tree.export_graphviz(clf, out_file=None, feature_names=list(X),
                                 class_names=['Died', 'Survived'], filled=True))
graph.render()
tree.plot_tree(clf)
plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

