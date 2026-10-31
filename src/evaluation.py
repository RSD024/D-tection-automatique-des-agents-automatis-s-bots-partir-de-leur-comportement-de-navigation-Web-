
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# 1. Charger les données
df = pd.read_csv("data/bot_sessions.csv")
df = df.drop(columns=["profile", "distinct_user_agents"])

X = df.drop(columns=["label"])
y = df["label"]

# 2. Refaire la même séparation
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Normaliser
scaler = joblib.load("models/scaler.pkl")
X_test = scaler.transform(X_test)

# 4. Charger et évaluer les modèles
model_files = {
    "Decision Tree": "models/decision_tree.pkl",
    "Random Forest": "models/random_forest.pkl",
    "KNN": "models/knn.pkl",
    "Logistic Regression": "models/logistic_regression.pkl",
    "Naive Bayes": "models/naive_bayes.pkl"
}

results = []

for name, filename in model_files.items():
    model = joblib.load(filename)
    y_pred = model.predict(X_test)

    results.append({
        "Modèle": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred)
    })

    print("\n==========", name, "==========")
    print(classification_report(y_test, y_pred))
    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))

# 5. Comparaison finale
results_df = pd.DataFrame(results)
print("\n===== COMPARAISON DES CINQ MODÈLES =====")
print(results_df.to_string(index=False))

# 6. Sauvegarder les résultats
results_df.to_csv("models/comparison_results.csv", index=False)
print("\nRésultats sauvegardés dans models/comparison_results.csv")
