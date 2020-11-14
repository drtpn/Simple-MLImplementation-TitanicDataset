#Titanic Dataset - Kaggle

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

rf=RandomForestClassifier(random_state=1)
lr=LogisticRegression(random_state=1)
nb=MultinomialNB()
dt=DecisionTreeClassifier(random_state=0)
gbm=GradientBoostingClassifier(n_estimators=10)

df=pd.read_csv('C:/Users/Lenovo PC/Desktop/VCET-STTP/ML/Titanic/train.csv')

df['Age'].fillna(df['Age'].mean(), inplace=True)

le=LabelEncoder()
le.fit(df['Sex'])
df['Sex']=le.transform(df['Sex'])

#print(df['Sex'].value_counts())
#print(df.isnull().sum())

x=df.drop('PassengerId', axis=1)
x=x.drop('Name', axis=1)
x=x.drop('Ticket', axis=1)
x=x.drop('Cabin', axis=1)
x=x.drop('Embarked', axis=1)
x=x.drop('Survived', axis=1)

y=df['Survived']

X_train,X_test,Y_train,Y_test=train_test_split(x,y,random_state=0,test_size=0.2)

rf.fit(X_train,Y_train)
lr.fit(X_train,Y_train)
dt.fit(X_train,Y_train)
nb.fit(X_train,Y_train)
gbm.fit(X_train,Y_train)

y_pred=dt.predict(X_test)
y_pred1=rf.predict(X_test)
y_pred2=nb.predict(X_test)
y_pred3=lr.predict(X_test)
y_pred4=gbm.predict(X_test)


print("Decision Tree: ",accuracy_score(Y_test,y_pred))
print("Random Forest: ",accuracy_score(Y_test,y_pred1))
print("Naive Bayes: ",accuracy_score(Y_test,y_pred2))
print("Logistic Regression: ",accuracy_score(Y_test,y_pred3))
print("GBM: ",accuracy_score(Y_test,y_pred4))
