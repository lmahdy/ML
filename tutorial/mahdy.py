import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('WeightHeight.csv')  # that means putting a copy of csv file in a data frame called df
print(df)  # print the data frame
print(df.isnull().any().any())  # check if there is any null value in the data frame

# Étape 1: Sélection de la variable prédictive
# X - La variable explicative: Un tableau 1D contenant:
# Toutes les lignes (les observations) et only the Height column
X = df['Height'].values
print(X)

# y - La variable expliquée: Un tableau 1D contenant:
# Toutes les lignes (les observations) et seulement la dernière colonne
y = df.iloc[:, -1].values
print(y)

# Visualisations des données sous forme de nuage de points
axes = plt.axes()
axes.grid()
plt.scatter(X, y, color='red')  # X et y représentent respectivement la taille et le poids
plt.xlabel('Height') # Height = taille en cm
plt.ylabel('Weight') # Weight = poids en kg
plt.title('Height vs Weight') # Height vs Weight = taille vs poids
plt.show() # Afficher le nuage de points
plt.close() # Fermer la fenêtre
