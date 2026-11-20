
import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression

# Charger modele1
train = pd.read_csv("data/dataClean/modele1/train.csv")

# Séparer les variables
X_train = train.drop(columns=["label"])
y_train = train["label"]

# Créer le modèle
model = LogisticRegression(max_iter=1000, random_state=42)

# Entraîner
model.fit(X_train, y_train)

# Sauvegarder
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/logistic_regression.pkl")

print("Logistic Regression entraînée et sauvegardée avec succès !")
