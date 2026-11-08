"""Sélection des variables : détection des colonnes à exclure et choix des features.

Reprend les décisions du notebook (section 01.7) :
- suppression des colonnes constantes ;
- suppression des colonnes redondantes (calculables à partir d'autres colonnes) ;
- `profile` n'est jamais une feature (il détermine la cible `label`).
"""
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_M1 = "label"      # modèle 1 : bot / non-bot
TARGET_M2 = "profile"    # modèle 2 : profil du bot (lignes label == 1)

# Redondances établies en 01.3 :
#   distinct_usernames == auth_failure_rate
#   auth_failure_count  = auth_failure_rate * request_count
#   path_repeat_ratio   = 1 - distinct_paths / request_count
REDUNDANT_COLUMNS = ["distinct_usernames", "auth_failure_count", "path_repeat_ratio"]


def detect_constant_columns(df: pd.DataFrame) -> list[str]:
    """Colonnes à variance nulle (une seule valeur, NaN compris)."""
    return [c for c in df.columns if df[c].nunique(dropna=False) <= 1]


def detect_integer_floats(df: pd.DataFrame) -> list[str]:
    """Colonnes float qui ne contiennent que des entiers (à convertir en int)."""
    return [c for c in df.select_dtypes("float").columns if (df[c] == df[c].round()).all()]


def get_columns_to_drop(df: pd.DataFrame) -> list[str]:
    """Colonnes constantes + redondantes présentes dans `df`."""
    redundant = [c for c in REDUNDANT_COLUMNS if c in df.columns]
    return detect_constant_columns(df) + redundant


def get_features(df_clean: pd.DataFrame) -> list[str]:
    """Toutes les colonnes sauf les cibles (`label`, `profile`)."""
    return [c for c in df_clean.columns if c not in (TARGET_M1, TARGET_M2)]


def high_correlation_pairs(df: pd.DataFrame, threshold: float = 0.95) -> pd.Series:
    """Paires de colonnes numériques avec |corrélation| >= threshold."""
    corr = df.select_dtypes("number").corr().abs()
    pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    return pairs[pairs >= threshold].sort_values(ascending=False)


def skew_stats(df: pd.DataFrame, exclude: list[str] | None = None) -> pd.DataFrame:
    """Statistiques d'échelle / asymétrie et candidates à log1p.

    log1p : uniquement les variables continues non bornées
    (|skew| > 1, min >= 0, max > 1, plus de 20 valeurs distinctes).
    """
    exclude = exclude or []
    num = df[[c for c in df.select_dtypes("number").columns if c not in exclude]]
    stats = pd.DataFrame({
        "min": num.min(),
        "max": num.max(),
        "std": num.std(),
        "skew": num.skew(),
    })
    stats["log1p_candidate"] = (
        (stats["skew"].abs() > 1) & (stats["min"] >= 0)
        & (stats["max"] > 1) & (num.nunique() > 20)
    )
    return stats


def get_log1p_columns(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    """Liste des colonnes candidates à la transformation log1p."""
    stats = skew_stats(df, exclude)
    return stats.index[stats["log1p_candidate"]].tolist()


def describe_model_inputs(df_raw: pd.DataFrame) -> dict:
    """Décisions de sélection pour les deux modèles, à partir du dataset brut."""
    to_drop = get_columns_to_drop(df_raw)
    int_cols = [c for c in detect_integer_floats(df_raw) if c not in to_drop]
    kept = [c for c in df_raw.columns if c not in to_drop]
    features = [c for c in kept if c not in (TARGET_M1, TARGET_M2)]
    return {
        "to_drop": to_drop,
        "int_columns": int_cols,
        "features": features,
        "model1": {"target": TARGET_M1, "exclude": to_drop + [TARGET_M2]},
        "model2": {"target": TARGET_M2, "exclude": to_drop + [TARGET_M1]},
        "log1p": get_log1p_columns(df_raw, exclude=to_drop + [TARGET_M1]),
    }


if __name__ == "__main__":
    raw = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "bot_sessions.csv")
    for key, value in describe_model_inputs(raw).items():
        print(f"{key:12s}: {value}")
