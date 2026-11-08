
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from feature_selection import (
    TARGET_M1,
    TARGET_M2,
    detect_integer_floats,
    get_columns_to_drop,
    get_features,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "bot_sessions.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

SEED = 42
TEST_SIZE = 0.2


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Charge le dataset brut."""
    return pd.read_csv(path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Applique les étapes 1 à 3 : colonnes inutiles, types, zéros négatifs."""
    cols_to_drop = get_columns_to_drop(df)
    cols_int = [c for c in detect_integer_floats(df) if c not in cols_to_drop]

    df_clean = df.drop(columns=cols_to_drop).copy()
    df_clean[cols_int] = df_clean[cols_int].astype(int)
    df_clean["path_entropy"] = df_clean["path_entropy"] + 0.0  # -0.0 -> 0.0
    return df_clean


def split_train_test(df_clean: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED):
    """Mélange + split stratifié sur le profil."""
    return train_test_split(
        df_clean,
        test_size=test_size,
        shuffle=True,
        stratify=df_clean[TARGET_M2],
        random_state=seed,
    )


def build_model_datasets(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    """Construit (train, test) pour chaque modèle ; la cible est la dernière colonne."""
    features = get_features(df_train)
    cols_m1 = features + [TARGET_M1]
    cols_m2 = features + [TARGET_M2]
    return {
        "modele1": (df_train[cols_m1], df_test[cols_m1]),
        "modele2": (
            df_train.loc[df_train[TARGET_M1] == 1, cols_m2],
            df_test.loc[df_test[TARGET_M1] == 1, cols_m2],
        ),
    }


def run_quality_checks(df_clean: pd.DataFrame, df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    """Contrôles qualité ; retourne la répartition des profils train / test."""
    assert df_clean.isna().sum().sum() == 0, "valeurs manquantes"
    assert not [c for c in df_clean.columns if df_clean[c].nunique() <= 1], "colonne constante"
    assert not (set(df_train.index) & set(df_test.index)), "fuite : lignes communes train/test"
    assert len(df_train) + len(df_test) == len(df_clean), "lignes perdues"
    assert not ((df_clean["path_entropy"] == 0) & np.signbit(df_clean["path_entropy"])).any(), "-0.0 restant"

    prop = pd.DataFrame({
        "train_%": df_train[TARGET_M2].value_counts(normalize=True) * 100,
        "test_%": df_test[TARGET_M2].value_counts(normalize=True) * 100,
    }).round(2)
    assert (prop["train_%"] - prop["test_%"]).abs().max() < 0.1, "stratification imparfaite"
    return prop


def save_datasets(datasets: dict, out_dir: Path = PROCESSED_DIR) -> None:
    """Écrit {out_dir}/{modele}/train.csv et test.csv, puis relit pour vérifier."""
    for name, (train, test) in datasets.items():
        (out_dir / name).mkdir(parents=True, exist_ok=True)
        train.to_csv(out_dir / name / "train.csv", index=False)
        test.to_csv(out_dir / name / "test.csv", index=False)
        print(f"{name} : train {train.shape} | test {test.shape}")

    print("\nVérification :")
    for name in datasets:
        for part in ("train", "test"):
            d = pd.read_csv(out_dir / name / f"{part}.csv")
            target = d.columns[-1]
            assert d.isna().sum().sum() == 0, f"valeurs manquantes dans {name}/{part}"
            print(f"  {name}/{part}.csv : {d.shape[0]} lignes, {d.shape[1]} colonnes, "
                  f"cible = '{target}', classes = {sorted(d[target].unique().tolist())}")


def load_model_data(model: str, part: str, base_dir: Path = PROCESSED_DIR):
    """Charge X, y d'un modèle ('modele1' | 'modele2') pour 'train' ou 'test'."""
    d = pd.read_csv(base_dir / model / f"{part}.csv")
    return d.iloc[:, :-1], d.iloc[:, -1]


def main() -> None:
    df = load_raw()
    df_clean = clean_dataset(df)
    df_train, df_test = split_train_test(df_clean)
    prop = run_quality_checks(df_clean, df_train, df_test)

    print("df_clean :", df_clean.shape, "| colonnes supprimées :", get_columns_to_drop(df))
    print("df_train :", df_train.shape, "| df_test :", df_test.shape)
    print("\nRépartition des profils (%) :\n", prop)
    print("\nRépartition du label (%) :")
    print(pd.DataFrame({
        "train": df_train[TARGET_M1].value_counts(normalize=True) * 100,
        "test": df_test[TARGET_M1].value_counts(normalize=True) * 100,
    }).round(2))
    print()
    save_datasets(build_model_datasets(df_train, df_test))


if __name__ == "__main__":
    main()
