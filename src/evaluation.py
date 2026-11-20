
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# 1. Charger les données de test
test = pd.read_csv("data/dataClean/modele1/test.csv")

X_test = test.drop(columns=["label"])
y_test = test["label"]

# 2. Charger les modèles sauvegardés
models = {
    "Decision Tree": "models/decision_tree.pkl",
    "Random Forest": "models/random_forest.pkl",
    "KNN": "models/knn.pkl",
    "Logistic Regression": "models/logistic_regression.pkl",
    "Naive Bayes": "models/naive_bayes.pkl"
}

# 3. Créer le dossier des résultats
os.makedirs("results", exist_ok=True)

results = []

# 4. Évaluer chaque modèle
for name, path in models.items():

    model = joblib.load(path)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot(cmap="Blues")
    plt.title(name)
    plt.savefig(f"results/{name.replace(' ', '_')}_confusion_matrix.png")
    plt.close()

    # Rapport de classification
    print(f"\n===== {name} =====")
    print(classification_report(y_test, y_pred, zero_division=0))

# 5. Comparer les modèles
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="F1-score", ascending=False)

print("\n===== COMPARAISON DES MODELES =====")
print(results_df.to_string(index=False))

results_df.to_csv("results/evaluation_results.csv", index=False)

print("\nRésultats sauvegardés dans le dossier results.")
