import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
df = pd.read_csv("data/bot_sessions.csv")

# Supprimer la colonne profile
df = df.drop(columns=["profile", "distinct_user_agents"])

# Calculer les corrélations
correlation = df.corr()

# Afficher les corrélations avec label
print("Corrélation avec la cible :")
print(correlation["label"].sort_values(ascending=False))

# Visualiser les corrélations
plt.figure(figsize=(12, 8))
sns.heatmap(correlation, cmap="coolwarm", center=0)
plt.title("Matrice de corrélation")
plt.tight_layout()
plt.show()
from sklearn.ensemble import RandomForestClassifier

# Préparer les données
X = df.drop(columns=["label"])
y = df["label"]

# Entraîner Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

# Calculer l'importance des variables
importance = pd.DataFrame({
    "Variable": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance", ascending=False
)

print("\nImportance des variables :")
print(importance)

# Visualiser les importances
plt.figure(figsize=(10, 6))
sns.barplot(data=importance, x="Importance", y="Variable")
plt.title("Importance des variables - Random Forest")
plt.tight_layout()
plt.show()
