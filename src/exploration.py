import pandas as pd

# Charger le dataset
df = pd.read_csv("data/bot_sessions.csv")

# Afficher les dimensions
print("Dimensions du dataset :", df.shape)

# Afficher les noms des colonnes
print("\nColonnes :")
print(df.columns.tolist())

# Afficher les 5 premières lignes
print("\nPremières lignes :")
print(df.head())

# Afficher les types de données
print("\nTypes des données :")
print(df.dtypes)

# Afficher les valeurs manquantes
print("\nValeurs manquantes :")
print(df.isnull().sum())

# Afficher les informations générales
print("\nInformations générales :")
df.info()