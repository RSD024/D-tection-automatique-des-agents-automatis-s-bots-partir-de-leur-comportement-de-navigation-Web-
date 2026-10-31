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
# Vérifier les doublons
print("\nNombre de doublons :")
print(df.duplicated().sum())

# Statistiques descriptives
print("\nStatistiques descriptives :")
print(df.describe())

# Répartition des classes
print("\nRépartition Humain / Bot :")
print(df["label"].value_counts())
print("\nRelation entre les variables et la cible :")
print(df.corr(numeric_only=True)["label"].sort_values(ascending=False))
print("\nVariables constantes :")
print(df.nunique()[df.nunique() == 1])

print("\nNombre de doublons complets :")
print(df.duplicated().sum())
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/bot_sessions.csv")
df = df.drop(columns=["profile", "distinct_user_agents"])

X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nNombre de lignes communes entre train et test :",
      len(set(map(tuple, X_train.values)) & set(map(tuple, X_test.values))))