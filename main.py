
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv('Final_dataset.csv')

print(dataset.head())

print(dataset.info())

#correlation data
f,ax=plt.subplots(figsize=(14,14))
sns.heatmap(dataset.corr(),annot=True,ax=ax,fmt=".2f")
plt.xticks(rotation=90)
# plt.show()

dataset.hist(bins=50, figsize=(28,28))
# plt.show()

y =  dataset.status.values

X = dataset.drop(['Name', 'status'], axis=1)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)


knn = KNeighborsClassifier(n_neighbors=6)


knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print('Test set prediction: \n {}'.format(y_pred))


knn.score(X_test, y_test) * 100
print(knn.score(X_test, y_test)*100)
print(X_test)
print(y_test)

# user_input=input("Enter all 16 values separated by coma")
# user_input=list(map(float, user_input.split(',')))
user_input = [119.992,0.00784,0.0037,0.00554,0.01109,0.04374,0.426,0.02182,0.0313,0.02971,0.06545,0.02211,21.033]


y_pred = knn.predict([user_input])

print(y_pred)

if y_pred==1:
    print("Person is affected with Parkinson")
else:
    print("Person is not affected with parkinson")


import pickle

filename = 'parkinsons_detection_model.sav'
pickle.dump(knn, open(filename , 'wb'))

# loaded_model = pickle.load(open('parkinsons_detection_model.sav', 'rb'))
