import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

parkinsons_data = pd.read_csv('Parkinsons_Data_updated.csv')


# parkinsons_data.head()

# parkinsons_data.shape
# parkinsons_data.info()

# parkinsons_data.isnull().sum()

# parkinsons_data.describe()

# parkinsons_data['status'].value_counts()

# parkinsons_data.groupby('status').mean()

X = parkinsons_data.drop(columns=['name','status'], axis=1)
Y = parkinsons_data['status']

# print(X)

# print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# print(X.shape, X_train.shape, X_test.shape)

# scaler = StandardScaler()

# scaler.fit(X_train)




# X_train = scaler.transform(X_train)

# X_test = scaler.transform(X_test)


# print(X_train)

# Model Training

# Support Vector Machine Model

model = svm.SVC(kernel='linear')

model.fit(X_train, Y_train)


# accuracy score on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)


print('Accuracy score of training data : ', training_data_accuracy)

# accuracy score on training data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)







print('Accuracy score of test data : ', test_data_accuracy)

# not have
input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500)
# input_data = (128.001,138.052,122.08,0.00436,0.00003,0.00137,0.00166,0.00411,0.02297,0.21,0.01323,0.01072,0.01677,0.03969,0.00481,24.692)

# have
# input_data = (119.992,157.302,74.997,0.00784,0.00007,0.0037,0.00554,0.01109,0.04374,0.426,0.02182,0.0313,0.02971,0.06545,0.02211,21.033)


# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the data
# std_data = scaler.transform(input_data_reshaped)

prediction = model.predict(input_data_reshaped)
print(prediction)


if (prediction[0] == 0):
  print("The Person does not have Parkinsons Disease")

else:
  print("The Person has Parkinsons")

filename = 'parkinsons_model.sav'
pickle.dump(model, open(filename , 'wb'))

loaded_model = pickle.load(open('parkinsons_model.sav', 'rb'))

# for column in X.columns:
#   print(column)
