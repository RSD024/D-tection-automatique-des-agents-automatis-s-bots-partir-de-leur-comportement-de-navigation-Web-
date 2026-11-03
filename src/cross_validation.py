
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# 1. Charger les données
df = pd.read_csv("data/bot_sessions.csv")
df = df.drop(columns=["profile", "distinct_user_agents"])

X = df.drop(columns=["label"])
y = df["label"]

# 2. Définir les modèles
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB()
}

# 3. Définir la validation croisée
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4. Évaluer les modèles
results = []

for name, model in models.items():

    # Normalisation à l'intérieur de chaque fold
    pipeline = make_pipeline(StandardScaler(), model)

    scores = cross_val_score(
        pipeline, X, y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    results.append({
        "Modèle": name,
        "Accuracy moyenne": scores.mean(),
        "Écart-type": scores.std()
    })

    print("\n", name)
    print("Scores :", scores)
    print("Accuracy moyenne :", round(scores.mean(), 4))
    print("Écart-type :", round(scores.std(), 4))

# 5. Comparaison finale
results_df = pd.DataFrame(results)

print("\n===== RESULTATS CROSS-VALIDATION =====")
print(results_df.to_string(index=False))

# 6. Sauvegarder les résultats
results_df.to_csv("models/cross_validation_results.csv", index=False)

print("\nRésultats sauvegardés avec succès !")
