# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('WeightHeight.csv')

# Check the first few rows of the dataframe to understand its structure
print(df.head())

# Assuming 'Height' is the independent variable (X) and 'Weight' is the dependent variable (y)
X = df[['Height']].values
y = df['Weight'].values

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a linear regressor
regressor = LinearRegression()# Create and train a linear regressor in the figure its the line of regression
regressor.fit(X_train, y_train)# Create and train a linear regressor in the figure its the line of regression
#the difference regressor and regrossor.fit is that the first one is the line of regression and the second one is the line of regression with the data of training

# Plotting the data
plt.scatter(X_train, y_train, color='blue', label='Données d\'entraînement')# means the data of training in the figure its the blue points
plt.scatter(X_test, y_test, color='red', label='Données de test')# means the data of testing in the figure its the red points
plt.plot(X_train, regressor.predict(X_train), color='black', label='Ligne de régression')# means the line of regression in the figure its the black line

plt.title('Régression Linéaire du Poids par Rapport à la Taille')
plt.xlabel('Taille (en cm ou en inch selon les données)')
plt.ylabel('Poids (en kg ou en lbs selon les données)')
plt.legend()
plt.show()# show the figure
