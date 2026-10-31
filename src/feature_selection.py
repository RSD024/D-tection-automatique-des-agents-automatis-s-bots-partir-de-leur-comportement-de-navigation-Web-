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
print("Valeurs différentes :", df["distinct_user_agents"].nunique())