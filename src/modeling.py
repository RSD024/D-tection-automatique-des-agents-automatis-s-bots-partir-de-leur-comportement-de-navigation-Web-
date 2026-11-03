
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. Charger les données
df = pd.read_csv("data/bot_sessions.csv")

# 2. Supprimer les colonnes inutiles
df = df.drop(columns=["profile", "distinct_user_agents"])

# 3. Séparer X et y
X = df.drop(columns=["label"])
y = df["label"]

# 4. Séparer les données
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Normaliser les données
scaler = joblib.load("models/scaler.pkl")
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# 6. Définir les cinq modèles
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB()
}

# 7. Entraîner et sauvegarder les modèles
for name, model in models.items():
    print(f"\nEntraînement : {name}")

    model.fit(X_train, y_train)

    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, "models/" + filename)

    print("Modèle sauvegardé :", filename)

print("\nEntraînement des cinq modèles terminé !")
