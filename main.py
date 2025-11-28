import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetes_dataset = pd.read_csv('diabetes.csv')
describing = diabetes_dataset.describe()

value_counts = diabetes_dataset['Outcome'].value_counts()
print(value_counts)

grouped_data = diabetes_dataset.groupby('Outcome').mean()
print(grouped_data)

# Separating data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis = 1)
Y = diabetes_dataset['Outcome']

# Standardization of Data
scalar = StandardScaler()
scalar.fit(X)
standardized_data = scalar.transform(X)
print(standardized_data)
X = standardized_data
Y = diabetes_dataset['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Model Training
classifier = svm.SVC(kernel = 'linear')
classifier.fit(X_train,Y_train)

# Accuracy Score on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print(f'Accuracy Score of training data = {training_data_accuracy}')

# Accuracy Score on test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print(f'Accuracy Score of test data = {test_data_accuracy}')

input_data = (6,148,72,35,0,33.6,0.627,50)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

std_data = scalar.transform(input_data_reshaped)
prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')
