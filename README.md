# Prédiction du diabète à l'aide d'un modèle de Machine Learning

Le diabète est une maladie chronique qui touche des millions de personnes dans le monde. La détection précoce est essentielle pour améliorer la prise en charge. L’objectif ici est de tester différentes approches de machine learning pour créer un modèle prédictif fiable.

Ce projet comprend: 
- le nettoyage et la préparation des données,
- l'entraînement de plusieurs modèles de machine learning,
- l'évaluation de leurs performances.

## Le jeu de données

Le dataset utilisé contient **768 observations** de femmes indiennes, avec **8 caractéristiques médicales** :

- `Pregnancies`: Nombre de grossesses
- `Glucose`: Niveau de glucose
- `BloodPressure`: Pression artérielle diastolique
- `SkinThickness`: Épaisseur de la peau
- `Insulin`: Niveau d'insuline
- `BMI`: Indice de masse corporelle
- `DiabetesPedigreeFunction`: Antécédents familiaux
- `Age`: Âge

La **variable cible** est `Outcome` :
- `1`: La personne est diabétique
- `0`: La personne ne l'est pas

## Les technologies utilisées

- Python
- Pandas / NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebook

## Le traitement des valeurs manquantes

Dans ce jeu de données, certaines variables contiennent des valeurs aberrantes comme des zéros, qui ne sont pas réalistes d’un point de vue médical. Par exemple, un taux de glucose de 0, un taux d'insuline de 0, ou une pression artérielle à 0 ne sont des cas plausibles que chez les patients inconscients ou morts.
Ces zéros sont donc interprétés comme des valeurs manquantes et ont été traités en plusieurs étapes :
- **Suppression de certaines caractéristiques:**
Certaines colonnes présentaient un taux de valeurs incohérentes trop élevé. Ces variables ont été supprimées du dataset pour ne pas introduire de biais dans la modélisation.
- **mputation des valeurs manquantes restantes:**
Pour les colonnes conservées, les valeurs aberrantes ont été remplacées à l'aide d'une méthode d'imputation par K plus proches voisins (KNN imputer). Cette méthode estime les valeurs manquantes à partir des données les plus proches (similaires) dans le dataset.

Une imputation par la médiane a également été testée, mais les résultats obtenus avec cette méthode étaient moins performants, car elle ne tient pas compte des relations entre les variables.
L’imputation KNN, en s’appuyant sur la structure locale des données, a permis d’obtenir de meilleures performances de prédiction sur les modèles de machine learning testés par la suite.

## Les différents modèles testés et leurs performances

Trois modèles ont été testés: 
- Random Forest (Forêt aléatoire),
- Neural Network (Réseau de neurones),
- Logistic Regression (Régression logistique). 

Les hyperparamètres de chacun des modèles ont été réglés lors de l'entraînement à l'aide de la méthode `RandomizedSearchCV` disponible via `scikit-learn`. Cette méthode permet de tester aléatoirement diverses combinaisons d'hyperparamètres et de sélectionner la meilleure d'entre elles.

Ensuite, l'ensemble de validation a été utilisé pour choisir le meilleur modèle. Lors de cette simulation, ce fut le modèle Random Forest avec une accuracy = 0.78%, un F1-score = 68.4 et un recall = 67.5%.

Enfin, les performances de ce modèle ont été testées sur l'ensemble de test. 
Sur l'ensemble de test, l'accuracy est égale à 77.59%, ce qui est vraiment satisfaisant.

En revanche, ce qui est important en médecine c'est de minimiser le taux de faux négatif.
En effet, ne pas détecter le diabète présent chez un patient serait très embêtant.
C'est pourquoi il est important de se baser, également, sur le recall pour valider le modèle.

Ici le recall est à 63.41%, ce qui est insuffisant pour dire que ce modèle est suffisamment performant.
Cela était déjà visible lors de l'étude du pairplot.
En effet, sur ce dernier nous pouvions observer que les variables de ce jeu de données ne nous permettent pas de séparer nos classes en deux groupes distincts.

Afin de palier ce problème, il faudrait créer de nouvelles variables soit de nouvelles données de terrains significatives, soit combiner certaines variables ou inclure des variables d'interaction.