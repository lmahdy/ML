import pandas as pd
from sklearn.tree import DecisionTreeClassifier#it will import the decision tree classifier
from sklearn.model_selection import train_test_split#it will import the train test split function
from sklearn.metrics import accuracy_score#it will import the accuracy score function
import joblib#it will import the joblib function
from sklearn import tree#it will import the tree function
music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])#it will create a new data frame without the genre column and it will return a new data frame
#print(X)
y = music_data['genre']#it will create a new data frame with only the genre column and it will return a new data frame
# #print(y)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)#it will split the data into two parts one for training and one for testing 
model = DecisionTreeClassifier()
#model.fit(X_train, y_train)#it will train the model
model.fit(X, y)#it will train the model
tree.export_graphviz(model, out_file='music-recommender.dot',
feature_names=['age','gender'],class_names=sorted(y.unique()),label='all',rounded=True,filled=True)#it will create a dot file which will contain the decision tree
# model = joblib.load('music-recommender.joblib')#it will save the model in a file
# predictions = model.predict([[21,1]])#it will predict the output
# print(predictions)
##predictions = model.predict(X_test)#it will predict the output
#to calculate the accuracy of the model we compare the predictions with the y_test means it will compare the predicted output with the actual output

##score = accuracy_score(y_test, predictions)#it will give the accuracy of the model
##print(score)