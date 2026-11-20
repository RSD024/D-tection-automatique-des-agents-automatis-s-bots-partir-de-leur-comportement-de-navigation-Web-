import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

# Charger le dataset modele1
train = pd.read_csv("data/dataClean/modele1/train.csv")

# Séparer les variables
X_train = train.drop(columns=["label"])
y_train = train["label"]

# Créer le modèle
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Entraîner
model.fit(X_train, y_train)

# Sauvegarder
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/random_forest.pkl")

print("Random Forest entraîné et sauvegardé avec succès !")