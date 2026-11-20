
import pandas as pd
import joblib
import os
from sklearn.neighbors import KNeighborsClassifier

# Charger modele1
train = pd.read_csv("data/dataClean/modele1/train.csv")

# Séparer les variables
X_train = train.drop(columns=["label"])
y_train = train["label"]

# Créer le modèle
model = KNeighborsClassifier(n_neighbors=5)

# Entraîner
model.fit(X_train, y_train)

# Sauvegarder
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/knn.pkl")

print("KNN entraîné et sauvegardé avec succès !")
