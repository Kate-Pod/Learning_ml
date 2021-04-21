#Будем использовать датасет по оттоку клиентов телеком-оператора с сайта: https://bigml.com/user/francisco/gallery/dataset/5163ad540c0b5e5b22000383
import pandas as pd
import numpy as np
import os

#Назначим рабочую директорию
os.chdir("C:\\Users\\Администратор.WIN-U1NLG8MM702\\Documents\\Katerina\\Datasets for analysis")

#Загрузка и знакомство с данными

df=pd.read_csv('telecom_churn.csv')
df.head() #посмотрим на первые 5 строк
df.columns
#Целевая переменная: Churn – бинарный признак оттока (1 – потеря клиента - отток)
df.churn.unique() #array([False,  True])
df.shape #3333 строки и 20 столбцов
print(df.info()) #общая инфо по датасету


df['churn'] = df['churn'].astype('int64') #приведем к типу int64 [0,1]
df.describe() #статистики по всем числовым переменным
df.describe(include=['object', 'bool']) #статистики по нечисловым переменным
df['churn'].value_counts() #распределение данных

#2850 пользователей из 3333 — лояльные, значение переменной Churn у них — 0
df['churn'].value_counts() #распределение данных
df['area code'].value_counts(normalize=True) #распределение пользователей по переменной Area code.
# normalize=True, чтобы посмотреть относительные частоты.


#Сортировка
df.sort_values(by='total day charge',ascending=False).head()

#Сортировка по группе столбцов
df.sort_values(by=['churn', 'total day charge'], ascending=[True, False]).head()


#Индексация и извлечение данных
#Какова доля людей нелояльных пользователей в нашем датафрейме?
df['сhurn'].mean() #14,5%


df[P(df['Name'])] - DF, состоящий только из строк, удовлетворяющих условию P по столбцу Name
#Каковы средние значения числовых признаков среди нелояльных пользователей?
df[df['churn'] == 1].mean()
 
#Сколько в среднем в течение дня разговаривают по телефону нелояльные пользователи?
df[df['churn'] == 1]['total day minutes'].mean() # 206.91407867494823
#Какова максимальная длина международных звонков среди лояльных пользователей (Churn == 0), не пользующихся услугой международного роуминга ('International plan' == 'No')?
df[(df['churn'] == 0) & (df['international plan'] == 'No')]['total intl minutes'].max() # 18.899999999999999

#Для индексации по названию используется метод loc, по номеру — iloc (в случае с loc учитываются и начало, и конец слайса).
df.loc[0:5, 'state':'area code'] # значения для id строк от 0 до 5 и для столбцов от State до Area code

df.iloc[0:5, 0:3] #первые 5 строк и 3 столбца
df[:1] или df[-1:] #первая или последняя строчка датафрейма


#Применение функций к ячейкам, столбцам и строкам
df.apply(np.max) # Применение функции к каждому столбцу (axis=1 – для строк)
d = {'no' : False, 'yes' : True}
df['international plan'] = df['international plan'].map(d) 
# метод map можно использовать для замены значений в колонке, передав ему в качестве аргумента словарь вида {old_value: new_value}:

df = df.replace({'voice mail plan': d}) # Аналогично с replace

 

#Группировка данных
df.groupby(by=grouping_columns)[columns_to_show].function() – вид такой

#Группирование данных по Churn и вывод статистик по 3 столбцам в каждой группе.
columns_to_show = ['total day minutes', 'total eve minutes', 'total night minutes']
df.groupby(['churn'])[columns_to_show].describe(percentiles=[])

#То же самое, но по-другому: передав в agg список функций:
columns_to_show = ['total day minutes', 'total eve minutes', 'total night minutes']

df.groupby(['churn'])[columns_to_show].agg([np.mean, np.std, np.min, np.max])

#Сводные таблицы
pd.crosstab(df['churn'], df['international plan']) #таблица сопряженности
 
'''В Pandas за сводные таблицы отвечает метод pivot_table, который принимает в качестве параметров:
•	values – список переменных, по которым требуется рассчитать нужные статистики
•	index – список переменных, по которым нужно сгруппировать данные
•	aggfunc — то, что нам нужно посчитать по группам — сумму, среднее и т.п.'''

df.pivot_table(['total day calls', 'total eve calls', 'total night calls'],
['area code'], aggfunc='mean').head(10)
 
#Предобработка данных

total_calls = df['total day calls'] + df['total eve calls'] + \
                  df['total night calls'] + df['total intl calls']

#Создание столбца
df.insert(loc=len(df.columns), column='total calls', value=total_calls)
# loc - номер столбца, после которого нужно вставить данный Series
# мы указали len(df.columns), чтобы вставить его в самом конце

#ИЛИ ТАК:
df['total charge'] = df['total day charge'] + df['total eve charge'] + df['total night charge'] + df['total intl charge']

#Удаление столбцов или строк
df = df.drop(['total charge', 'total calls'], axis=1) #(0 – для строк)
df.drop([1, 2]).head() # а вот так можно удалить строчки


#Анализ и прогнозирование
#Посмотрим, как отток связан с признаком "Подключение международного роуминга" (International plan)
pd.crosstab(df['churn'], df['international plan'], margins=True)
 
#Когда роуминг подключен, доля оттока намного выше (137/323 vs 346/3010).
Посмотрим на еще один важный признак – "Число обращений в сервисный центр" (Customer service calls).
pd.crosstab(df['churn'], df['customer service calls'], margins=True)
 
#Доля оттока сильно возрастает начиная с 4 звонков в сервисный центр (26/66).
#Добавим теперь в наш DataFrame бинарный признак — результат сравнения Customer service calls > 3.
df['many_service_calls'] = (df['customer service calls'] > 3).astype('int') #true false превратили в 0,1

pd.crosstab(df['many_service_calls'], df['churn'], margins=True)
 
#Интерпретация: где количество звонков меньше 3 и где нет оттока – 2721 случаев. 
#Объединим рассмотренные выше условия:
pd.crosstab(df['many_service_calls'] & df['international plan'] , df['churn'])
 
'''9 случаев – звонков больше 3 и подключен роуминг, оттока нет. 19 – то же, но отток есть.
Вывод для дальнейшего прогноза
Доля лояльных клиентов в выборке – 85.5% ( (2841+9) / (2841+9+464+19) ). 
То есть доли правильных ответов (accuracy) последующих моделей должны быть как минимум не меньше, а лучше, значительно выше этой цифры.'''
