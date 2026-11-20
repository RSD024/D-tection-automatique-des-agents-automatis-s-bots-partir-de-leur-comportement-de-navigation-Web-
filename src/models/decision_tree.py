import pandas as pd
import joblib
import os
from sklearn.tree import DecisionTreeClassifier

# Charger les données de modele1
train = pd.read_csv("data/dataClean/modele1/train.csv")

# Séparer les variables et la cible
X_train = train.drop(columns=["label"])
y_train = train["label"]

# Créer le modèle
model = DecisionTreeClassifier(random_state=42)

# Entraîner le modèle
model.fit(X_train, y_train)

# Sauvegarder le modèle
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/decision_tree.pkl")