
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from feature_selection import (
    REDUNDANT_COLUMNS,
    describe_model_inputs,
    detect_constant_columns,
    detect_integer_floats,
    high_correlation_pairs,
    skew_stats,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "bot_sessions.csv"
FIGURES_DIR = ROOT / "results" / "figures"

RATE_COLUMNS = ["path_repeat_ratio", "static_asset_ratio", "error_rate",
                "not_found_rate", "auth_failure_rate"]
KEY_COLUMNS = ["request_count", "requests_per_minute", "interval_cv",
               "distinct_paths", "path_entropy"]

sns.set_theme(style="whitegrid")


def title(text: str) -> None:
    print("\n" + "_" * 60)
    print(text)
    print("_" * 60)


def save_figure(name: str, show: bool = False) -> None:
    """Enregistre la figure courante dans results/figures/ (et l'affiche si show)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{name}.png", dpi=150)
    if show:
        plt.show()
    plt.close()


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


# --------------------------------------------------------------------------
# 00 - Visualisation du dataset
# --------------------------------------------------------------------------
def overview(df: pd.DataFrame) -> None:
    title("1-DIMENSIONS DU DATASET")
    print(f"Nombre de lignes   : {df.shape[0]}")
    print(f"Nombre de colonnes : {df.shape[1]}")

    title("2-INFORMATIONS SUR LES COLONNES")
    print(pd.DataFrame({
        "Type": df.dtypes,
        "Valeurs_non_nulles": df.count(),
        "Valeurs_uniques": df.nunique(),
        "Valeurs_manquantes": df.isnull().sum(),
    }))

    title("3-APERCU DES DONNEES")
    print("PREMIERES LIGNES");  print(df.head())
    print("\nDERNIERES LIGNES"); print(df.tail())

    title("4-STATISTIQUES DESCRIPTIVES")
    print(df.describe(include="all").T)

    title("5-VALEURS MANQUANTES")
    print(pd.DataFrame({
        "Nombre": df.isnull().sum(),
        "Pourcentage (%)": (df.isnull().mean() * 100).round(2),
    }).sort_values("Nombre", ascending=False))

    title("6-DOUBLONS")
    print(f"Nombre de lignes dupliquées : {df.duplicated().sum()}")


def numeric_columns(df: pd.DataFrame) -> pd.Index:
    return df.select_dtypes(include=np.number).columns.drop("label", errors="ignore")


def plot_class_distribution(df: pd.DataFrame, show: bool = False) -> None:
    title("7-DISTRIBUTION DES CLASSES")
    print(df["label"].value_counts().to_frame("Effectif"))
    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x="label")
    plt.title("Distribution des classes")
    plt.xlabel("Label")
    plt.ylabel("Nombre de sessions")
    save_figure("distribution_classes", show)


def plot_profile_distribution(df: pd.DataFrame, show: bool = False) -> None:
    title("8-DISTRIBUTION DES PROFILS")
    print(df["profile"].value_counts().to_frame("Effectif"))
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, y="profile", order=df["profile"].value_counts().index)
    plt.title("Distribution des profils")
    plt.xlabel("Nombre de sessions")
    plt.ylabel("Profil")
    save_figure("distribution_profils", show)


def plot_numeric_distributions(df: pd.DataFrame, show: bool = False) -> None:
    df[numeric_columns(df)].hist(figsize=(18, 14), bins=30, edgecolor="black")
    plt.suptitle("Distribution des variables numériques", fontsize=16)
    save_figure("distribution_variables_numeriques", show)


def plot_correlation_matrix(df: pd.DataFrame, show: bool = False) -> None:
    title("9-MATRICE DE CORRELATION")
    cols = numeric_columns(df).tolist() + ["label"]
    plt.figure(figsize=(14, 10))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Matrice de corrélation")
    save_figure("matrice_correlation", show)


def summary(df: pd.DataFrame) -> None:
    title("--SYNTHESE :--")
    print(f"Observations          : {len(df)}")
    print(f"Variables             : {df.shape[1]}")
    print(f"Valeurs manquantes    : {df.isnull().sum().sum()}")
    print(f"Doublons              : {df.duplicated().sum()}")
    print(f"Colonnes numériques   : {len(numeric_columns(df)) + 1}")
    print(f"Colonnes catégorielles: {len(df.select_dtypes(include=['object', 'string']).columns)}")


# --------------------------------------------------------------------------
# 01 - Analyse du dataset
# --------------------------------------------------------------------------
def analyse_types(df: pd.DataFrame) -> None:
    """01.1 - Variables constantes, types réels, quasi-binaires, discrètes."""
    title("01.1-CONSTANTES, TYPES REELS, QUASI-BINAIRES")
    print("Colonnes constantes :", detect_constant_columns(df))
    print("\nFloat contenant uniquement des entiers (à convertir en int) :")
    print(detect_integer_floats(df))

    print("\nVariables à 2 valeurs :")
    for c in [c for c in df.columns if df[c].nunique() == 2]:
        print(f"  {c:20s} -> {sorted(df[c].unique().tolist())}")

    print("\nVariables discrètes (3 à 5 valeurs) :")
    for c in [c for c in df.select_dtypes("number").columns if 2 < df[c].nunique() <= 5]:
        print(f"  {c:20s} -> {sorted(df[c].unique().tolist())}")
    print(df.groupby("profile")["auth_failure_count"].describe().round(2))


def determine(data: pd.DataFrame, x: str, y: str) -> bool:
    """True si x -> y (aucune valeur de x associée à plusieurs valeurs de y)."""
    return bool((data.groupby(x)[y].nunique(dropna=False) <= 1).all())


def analyse_dependencies(df: pd.DataFrame) -> None:
    """01.2 - Dépendances fonctionnelles (x -> y)."""
    title("01.2-DEPENDANCES FONCTIONNELLES")
    for x, y in [("profile", "label"), ("profile", "auth_failure_rate")]:
        print(f"{x} -> {y} : {determine(df, x, y)}   |   {y} -> {x} : {determine(df, y, x)}")
    print("\n", pd.crosstab(df["profile"], df["label"]))
    print("\n", pd.crosstab(df["profile"], df["auth_failure_rate"]))

    constantes = detect_constant_columns(df)
    candidats = [c for c in df.columns if 1 < df[c].nunique() <= 50 and c not in constantes]
    print("\nScan automatique :")
    for x in candidats:
        for y in candidats:
            if x != y and determine(df, x, y):
                print(f"  {x} -> {y}")


def analyse_redundancy(df: pd.DataFrame, threshold: float = 0.95) -> None:
    """01.3 - Colonnes identiques, relations par formule, corrélations fortes."""
    title("01.3-REDONDANCES")
    constantes = detect_constant_columns(df)
    cols = [c for c in df.select_dtypes("number").columns if c not in constantes]

    print("Colonnes identiques :")
    found = False
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if df[a].equals(df[b]):
                print(f"  {a} == {b}")
                found = True
    if not found:
        print("  aucune")

    formules = {
        "auth_failure_count = auth_failure_rate x request_count":
            (df["auth_failure_count"], df["auth_failure_rate"] * df["request_count"]),
        "path_repeat_ratio = 1 - distinct_paths / request_count":
            (df["path_repeat_ratio"], 1 - df["distinct_paths"] / df["request_count"]),
    }
    print("\nRelations par formule :")
    for nom, (reel, calcule) in formules.items():
        ecart = float(np.abs(reel - calcule).max())
        print(f"  {nom:58s} écart max = {ecart:.2e} -> {'OUI' if ecart < 1e-9 else 'NON'}")

    print(f"\nPaires avec |corrélation| >= {threshold} :")
    print(high_correlation_pairs(df[cols], threshold).round(3).to_string())


def check_consistency(df: pd.DataFrame, eps: float = 1e-9) -> None:
    """01.4 - Cohérence logique et bornes."""
    title("01.4-COHERENCE LOGIQUE ET BORNES")
    verifications = {
        "taux hors [0, 1]": ((df[RATE_COLUMNS] < 0) | (df[RATE_COLUMNS] > 1)).any(axis=1).sum(),
        "request_count <= 0": (df["request_count"] <= 0).sum(),
        "requests_per_minute <= 0": (df["requests_per_minute"] <= 0).sum(),
        "interval_cv < 0": (df["interval_cv"] < 0).sum(),
        "distinct_paths > request_count": (df["distinct_paths"] > df["request_count"]).sum(),
        "auth_failure_count > request_count": (df["auth_failure_count"] > df["request_count"]).sum(),
        "error_rate < auth_failure_rate": (df["error_rate"] < df["auth_failure_rate"] - eps).sum(),
        "error_rate < not_found_rate": (df["error_rate"] < df["not_found_rate"] - eps).sum(),
        "error_rate < auth_failure_rate + not_found_rate":
            (df["error_rate"] < df["auth_failure_rate"] + df["not_found_rate"] - eps).sum(),
        "path_entropy < 0": (df["path_entropy"] < 0).sum(),
        "path_entropy > log2(distinct_paths)":
            (df["path_entropy"] > np.log2(df["distinct_paths"]) + eps).sum(),
        "distinct_paths == 1 et entropy != 0":
            ((df["distinct_paths"] == 1) & (df["path_entropy"].abs() > eps)).sum(),
        "distinct_depths > max_depth + 1": (df["distinct_depths"] > df["max_depth"] + 1).sum(),
    }
    resultat = pd.Series(verifications, name="violations").to_frame()
    resultat["statut"] = np.where(resultat["violations"] == 0, "OK", "ANOMALIE")
    print(resultat)

    zeros_neg = ((df["path_entropy"] == 0) & np.signbit(df["path_entropy"])).sum()
    print(f"\npath_entropy = -0.0 : {zeros_neg} lignes")
    print("\nLignes avec max_depth = 0 :")
    print(df[df["max_depth"] == 0].T)


def taux_outliers(s: pd.Series) -> float:
    """Pourcentage d'outliers selon la règle IQR."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean() * 100


def analyse_distributions(df: pd.DataFrame) -> None:
    """01.5 - Distributions par profil, bornes artificielles, outliers."""
    title("01.5-DISTRIBUTIONS PAR PROFIL ET OUTLIERS")
    print(df.groupby("profile")[KEY_COLUMNS].agg(["min", "median", "max"]).round(2))

    print("\nstatic_asset_ratio par profil :")
    print(df.groupby("profile")["static_asset_ratio"].agg(["min", "max"]).round(3))

    tableau = pd.DataFrame({
        "outliers_global_%": df[KEY_COLUMNS].apply(taux_outliers),
        "outliers_moyenne_par_profil_%": df.groupby("profile")[KEY_COLUMNS].agg(taux_outliers).mean(),
    }).round(1)
    print("\nOutliers IQR :")
    print(tableau)

    constantes = detect_constant_columns(df)
    features = [c for c in df.columns if c not in ["label", "profile"] + constantes]
    print("\nDoublons sur les features :", int(df.duplicated(subset=features).sum()))
    conflits = (df.groupby(features)["profile"].nunique() > 1).sum()
    print("Mêmes features avec profils différents :", int(conflits))


def analyse_skew(df: pd.DataFrame) -> None:
    """01.6 - Asymétrie et échelles."""
    title("01.6-ASYMETRIE ET ECHELLES")
    stats = skew_stats(df, exclude=detect_constant_columns(df) + ["label"])
    print(stats.round(3))
    print("\nCandidates à log1p :", stats.index[stats["log1p_candidate"]].tolist())
    print("Rapport d'échelle max/min des écarts-types :", round(stats["std"].max() / stats["std"].min(), 1))


def preprocessing_decisions(df: pd.DataFrame) -> None:
    """01.7 - Synthèse et décisions de preprocessing."""
    title("01.7-SYNTHESE ET DECISIONS DE PREPROCESSING")
    d = describe_model_inputs(df)
    print("MODELE 1 (bot / non-bot)")
    print("  cible     :", d["model1"]["target"])
    print("  à exclure :", d["model1"]["exclude"])
    print("  features  :", d["features"])
    print("\nMODELE 2 (profil du bot, lignes label == 1)")
    print("  cible     :", d["model2"]["target"])
    print("  à exclure :", d["model2"]["exclude"])
    print("\nTYPES : à convertir en int :", d["int_columns"])
    print("\nTRANSFORMATIONS")
    print("  log1p (modèles linéaires) :", d["log1p"])
    print("  normaliser path_entropy -0.0 -> 0.0")
    print("  standardisation : après split, sur train seulement")
    print("\nSPLIT : mélanger, puis split stratifié sur 'profile'")
    print("ÉVALUATION : F1 macro + matrice de confusion ; pipeline de bout en bout ;")
    print("  ablation avec/sans variables d'authentification ; baseline plat à 8 classes")


def main(show: bool = False) -> None:
    df = load_data()
    overview(df)
    plot_class_distribution(df, show)
    plot_profile_distribution(df, show)
    plot_numeric_distributions(df, show)
    plot_correlation_matrix(df, show)
    summary(df)

    analyse_types(df)
    analyse_dependencies(df)
    analyse_redundancy(df)
    check_consistency(df)
    analyse_distributions(df)
    analyse_skew(df)
    preprocessing_decisions(df)


if __name__ == "__main__":
    main()
