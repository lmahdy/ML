# Importation de la bibliothèque pandas pour la manipulation de données
import pandas as pd

# Importation de la bibliothèque seaborn pour la visualisation de données
import seaborn as sns

# Chargement des données depuis un fichier CSV dans un DataFrame pandas
diabetes = pd.read_csv("pima-indians-diabetes.csv")

# Renommage des colonnes pour une meilleure lisibilité
diabetes.columns=['NumTimesPrg','PlGlcConc','BloodP','SkinThick','TwoHourSerins','BMI','DiPedFunc','age','HasDiabetes']

# Vérification de la présence de valeurs manquantes dans les données
print(diabetes.isnull().any().any())

# Définition de différents groupes de caractéristiques pour expérimenter avec les modèles
feature1=['BMI']
feature2=['PlGlcConc','BloodP','age']
feature3=['NumTimesPrg','PlGlcConc','BloodP','SkinThick','TwoHourSerins','BMI','DiPedFunc','age']

# Sélection de la colonne cible pour la classification
y = diabetes['HasDiabetes']

# Sélection d'un ensemble de caractéristiques pour l'entrainement du modèle
x = diabetes[feature3]  # feature2 best

# Importation de la fonction pour diviser les données en ensembles d'entraînement et de test
from sklearn.model_selection import train_test_split

# Division des données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75)#train_size=0.75 means that 75% of the data is for training and 25% for testing
# Importation du modèle de régression logistique de sklearn
from sklearn.linear_model import LogisticRegression

# Création d'une instance du modèle de régression logistique
log = LogisticRegression()

# Entraînement du modèle avec l'ensemble d'entraînement
log.fit(x_train, y_train)

# Prédiction des résultats sur l'ensemble de test
y_pred = log.predict(x_test)
print(y_pred)
print("*************")
print(y_test)#difference between y_pred and y_test is that y_pred is the prediction of the model and y_test is the real data
# Importation des fonctions pour évaluer les performances du modèle
from sklearn import metrics

# Calcul de la matrice de confusion pour évaluer les erreurs de classification
confusionmatrix = metrics.confusion_matrix(y_test, y_pred)

# Affichage de la matrice de confusion
print(confusionmatrix)

# Calcul de différentes métriques pour évaluer la performance du modèle
ac1 = metrics.accuracy_score(y_test, y_pred)
pre1 = metrics.precision_score(y_test, y_pred)
rec1 = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
#give me the lines to affiche the courbe of the regression 
print("*************")
print(ac1,"ac1")
print("*************")
print(pre1,"pre1")
print("*************")
print(rec1,"rec1")
print("*************")
print(f1,"f1")
print("*************")

# Visualisation de la relation entre l'IMC et le diabète en utilisant une régression logistique
sns.regplot(x="BMI", y="HasDiabetes", data=diabetes, logistic=True)
import matplotlib.pyplot as plt

plt.show()
