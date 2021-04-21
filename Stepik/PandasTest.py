import pandas as pd
students =pd.read_csv('https://stepik.org/media/attachments/course/4852/StudentsPerformance.csv')
students.head() #первые 5 строк
students.describe() #основные описательные статистики

students.iloc[0:5,0:3] #выберем первые 5 строк и 3 столбца
students.iloc[[0,3,10],[-1,-2,-3]] 
students_with_names=students.iloc[[0,3,4,7,8]]
students_with_names.index=['Cercei','Tywin','Joffrey','Ilyn Payne','Gregor'] #дадим имена индексам
student_performance_with_names.loc =['Cercei', 'Joffrey'],['gender', 'writing score']] #выберем данные по индексам и столбцам


#У какой доли студентов из датасэта в колонке lunch указано free/reduced?
students.lunch.loc[students.lunch=='free/reduced'].count()/students.lunch.count()

#Сравним оценки студентов с урезанным ланчем и стандартным по 3 дисциплинам 
students[students.lunch=='standard'].describe()-students[students.lunch=='free/reduced'].describe()

#Переименуем названия столбцов для удобства работы с ними
students=students.rename(columns={'parental level of education': 'parental_level_of_education',
                                            'test preparation course': 'test_preparation_course',
                                            'math score': 'math_score',
                                            'reading score': 'reading_score',
                                            'writing score': 'writing_score'})
#Пробуем query фильтрацию
students.query("writing_score>74 & gender=='female'")
wrsc=80
students.query("writing_score > @wrsc")

#Отбор колонок/строк, содержащих определенные слова
students.filter(like='score',axis=1)

#Группировка и сортировка
students.groupby('gender').mean()
students.groupby('gender', as_index=False).aggregate({'math_score': ['mean', 'count', 'std'],'reading_score': ['std', 'min', 'max']})
students.sort_values(['gender', 'math_score'], ascending=True).groupby('gender').head(5)
#Создание и удаление колонок
students['total_score']=students.math_score+students.reading_score
students.drop(['total_score'],axis=1]

              
'''
В dataframe с именем my_stat сохранены данные с 4 колонками: session_value, group, time, n_users.  
В переменной session_value замените все пропущенные значения на нули.
В переменной n_users замените все отрицательные значения на медианное значение переменной n_users (без учета отрицательных значений, разумеется)              
'''    
my_stat=my_stat.fillna('0')
my_stat['session_value']=my_stat['session_value'].astype(float)
M = my_stat[my_stat.n_users >= 0.0].n_users.median()
my_stat.loc[my_stat['n_users'] < 0, 'n_users'] = M             

              
              
#Проанализируем набор данных о о действиях, которые совершают студенты на stepik
events=pd.read_csv('event_data_train.csv') #['step_id', 'timestamp', 'action', 'user_id']
events.head(10)
events.action.unique() #как distinct в sql  
events['date']=pd.to_datetime(events.timestamp, unit='s') #добавим столбец с датой в формате yyyy-mm-dd hh:mm:ss
events['day']=events.date.dt.date #yyyy-mm-dd    
events.pivot_table(index='user_id', columns='action', values='step_id', aggfunc='count', fill_value=0).head() #посмотрим, сколько студентов "discovered","passed","started", "viewed" степ
events[['user_id', 'day', ' timestamp']].drop_duplicates(subset=['user_id', 'day']).head() #удалим дубликаты в выделенных столбцах
