# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Import the required libraries.

Upload and read the dataset.

Check for any null values using the isnull() function.

From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Arunsamy D
RegisterNumber:  212224240016
*/
```
```python
import pandas as pd
data = pd.read_csv("/content/Employee.csv")
data.head()

print(data.isnull().sum())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['salary'] = le.fit_transform(data['salary'])

X = data[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours',
          'time_spend_company', 'Work_accident', 'promotion_last_5years', 'salary']]
y = data['left']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


sample_employee = [[0.5, 0.8, 9, 260, 6, 0, 1, 2]]  
prediction = dt.predict(sample_employee)
print(f"\nPrediction for sample employee (1=left, 0=stayed): {prediction[0]}")

```

## Output:

![image](https://github.com/user-attachments/assets/b666d16c-c314-4bab-a811-77a3c0b871a5)


![image](https://github.com/user-attachments/assets/bef733b7-cfc2-4286-9fa1-767770ca1331)


![image](https://github.com/user-attachments/assets/2b25b34e-f83a-412a-8ec6-8431da961d1b)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
