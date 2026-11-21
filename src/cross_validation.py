
import os
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import StratifiedKFold, cross_validate

# 1. Charger les données d'entraînement
train = pd.read_csv("data/dataClean/modele1/train.csv")

X_train = train.drop(columns=["label"])
y_train = train["label"]

# 2. Définir les cinq modèles
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "Naive Bayes": GaussianNB()
}

# 3. Définir la validation croisée
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision_weighted",
    "recall": "recall_weighted",
    "f1": "f1_weighted"
}

# 4. Évaluer les modèles
results = []

for name, model in models.items():

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    results.append({
        "Model": name,
        "Accuracy_mean": scores["test_accuracy"].mean(),
        "Accuracy_std": scores["test_accuracy"].std(),
        "Precision_mean": scores["test_precision"].mean(),
        "Precision_std": scores["test_precision"].std(),
        "Recall_mean": scores["test_recall"].mean(),
        "Recall_std": scores["test_recall"].std(),
        "F1_mean": scores["test_f1"].mean(),
        "F1_std": scores["test_f1"].std()
    })

# 5. Comparer les résultats
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="F1_mean",
    ascending=False
)

print("\n===== STRATIFIED 5-FOLD CROSS-VALIDATION =====")
print(results_df.to_string(index=False))

# 6. Sauvegarder les résultats
os.makedirs("results", exist_ok=True)

results_df.to_csv(
    "results/cross_validation_results.csv",
    index=False
)

print("\nRésultats sauvegardés dans results/cross_validation_results.csv")
