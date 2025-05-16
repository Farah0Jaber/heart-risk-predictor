import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv('Medicaldataset.csv') #this is the dataset that we are going to use

print("Dataset preview:") #viewing the first 5 rows of the dataset
print(data.head())
print("\nMissing values in each column:") #checking if there are any values missing
print(data.isnull().sum())

data['Result'] = data['Result'].map({'negative': 0, 'positive': 1}) #changing the result column to binary
X = data.drop(columns=['Result']) #input
y = data['Result'] #output

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #80% training and 20% testing
print("Data was successfully split.")

model = DecisionTreeClassifier() #making a classifier model descision tree
model.fit(X_train, y_train) #training the model using the provided data
predictions = model.predict(X_test) #after training the model, we are having it make predictions using the data

print("\nPredictions:") #displaying the prediction values
print(predictions)
print("\nActual:") #displaying the correct values
print(y_test.values)

accuracy = accuracy_score(y_test, predictions) #calculating how accurate the model is
print(f"\nModel accuracy: {accuracy}")

joblib.dump(model, 'heart_attack_model.joblib') #saving the model that we trained
