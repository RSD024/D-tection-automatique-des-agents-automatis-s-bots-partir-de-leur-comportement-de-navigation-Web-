import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Charger les données
df = pd.read_csv("data/bot_sessions.csv")

# 2. Supprimer la colonne profile
df = df.drop(columns=["profile"])

# 3. Séparer les caractéristiques (X) et la cible (y)
X = df.drop(columns=["label"])
y = df["label"]

# 4. Séparer les données en apprentissage et test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Normalisation des données
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Affichage des résultats
print("Nombre total de lignes :", len(df))
print("Nombre de variables :", X.shape[1])

print("\nTaille X_train :", X_train.shape)
print("Taille X_test  :", X_test.shape)

print("\nTaille y_train :", y_train.shape)
print("Taille y_test  :", y_test.shape)

print("\nRépartition de la cible :")
print(y.value_counts())
# Sauvegarder le scaler
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")

print("\nScaler sauvegardé avec succès !")