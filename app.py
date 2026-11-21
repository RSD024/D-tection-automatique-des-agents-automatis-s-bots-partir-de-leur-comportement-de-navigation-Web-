"""
Dashboard Streamlit — Détection & classification de sessions de bots
====================================================================
Modèle 1 : binaire (label 0/1)         -> train.csv / test.csv
Modèle 2 : multi-classe (profile)      -> train (1).csv / test (1).csv

Lancement :  streamlit run app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# --------------------------------------------------------------------------
# 0. CONFIGURATION
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Bot Sessions Dashboard",
    page_icon="🤖",
    layout="wide",
)

BASE = Path(__file__).resolve().parent
# Dossiers dans lesquels on cherche les fichiers (modifiable avec l'équipe)
SEARCH_DIRS = [
    BASE,
    BASE / "data",
    BASE / "models",
    BASE / "results",
    Path.cwd(),
]

# 11 variables d'entrée, ORDRE STRICT
FEATURES = [
    "request_count",
    "distinct_paths",
    "path_entropy",
    "static_asset_ratio",
    "error_rate",
    "not_found_rate",
    "auth_failure_rate",
    "requests_per_minute",
    "interval_cv",
    "max_depth",
    "distinct_depths",
]
INT_FEATURES = {
    "request_count",
    "distinct_paths",
    "max_depth",
    "distinct_depths",
}

# (min, max, défaut "réaliste") pour les widgets de prédiction
RANGES = {
    "request_count": (1, 300, 24),
    "distinct_paths": (1, 100, 8),
    "path_entropy": (0.0, 7.0, 2.85),
    "static_asset_ratio": (0.0, 1.0, 0.11),
    "error_rate": (0.0, 1.0, 0.0),
    "not_found_rate": (0.0, 1.0, 0.0),
    "auth_failure_rate": (0.0, 1.0, 0.0),
    "requests_per_minute": (0.0, 1000.0, 42.5),
    "interval_cv": (0.0, 3.0, 1.05),
    "max_depth": (0, 5, 3),
    "distinct_depths": (1, 6, 2),
}

PRESETS = {
    "🧑 Session type humaine": dict(
        request_count=20,
        distinct_paths=12,
        path_entropy=3.2,
        static_asset_ratio=0.35,
        error_rate=0.0,
        not_found_rate=0.0,
        auth_failure_rate=0.0,
        requests_per_minute=6.0,
        interval_cv=1.3,
        max_depth=3,
        distinct_depths=3,
    ),
    "🔎 Session type scanner": dict(
        request_count=80,
        distinct_paths=40,
        path_entropy=5.2,
        static_asset_ratio=0.0,
        error_rate=0.6,
        not_found_rate=0.55,
        auth_failure_rate=0.0,
        requests_per_minute=300.0,
        interval_cv=0.3,
        max_depth=3,
        distinct_depths=4,
    ),
    "🔐 Session type brute-force": dict(
        request_count=100,
        distinct_paths=2,
        path_entropy=0.5,
        static_asset_ratio=0.0,
        error_rate=0.9,
        not_found_rate=0.0,
        auth_failure_rate=0.9,
        requests_per_minute=400.0,
        interval_cv=0.05,
        max_depth=1,
        distinct_depths=1,
    ),
}

# Noms de fichiers attendus (le premier trouvé est utilisé)
FILES = {
    "raw": ["bot_sessions.csv"],
    "train_m1": ["train.csv"],
    "test_m1": ["test.csv"],
    "train_m2": ["train (1).csv", "train_1.csv"],
    "test_m2": ["test (1).csv", "test_1.csv"],
    "results": ["evaluation_results.csv", "results/evaluation_results.csv"],
}


# --------------------------------------------------------------------------
# 1. UTILITAIRES DE CHARGEMENT (robustes aux fichiers manquants)
# --------------------------------------------------------------------------
def find_file(names: list[str]) -> Path | None:
    """Cherche un fichier dans les dossiers usuels ; None si introuvable."""
    for d in SEARCH_DIRS:
        for n in names:
            p = d / n
            if p.is_file():
                return p
    return None


@st.cache_data(show_spinner=False)
def read_csv_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_csv_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))


def get_dataset(key: str) -> pd.DataFrame | None:
    """Charge un CSV : chemin auto -> sinon upload manuel -> sinon None."""
    up = st.session_state.get(f"upload_{key}")
    if up is not None:
        return read_csv_bytes(up.getvalue())
    p = find_file(FILES[key])
    return read_csv_path(str(p)) if p else None


@st.cache_resource(show_spinner=False)
def load_model_path(path: str):
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_model_bytes(data: bytes):
    return joblib.load(io.BytesIO(data))


def discover_models() -> list[Path]:
    found: list[Path] = []
    for d in SEARCH_DIRS:
        if d.is_dir():
            for ext in ("*.pkl", "*.joblib"):
                found += list(d.glob(ext))
    return list(dict.fromkeys(found))


def get_model_classes(model):
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "steps"):
        classes = getattr(model.steps[-1][1], "classes_", None)
    return classes


def model_task(model) -> str | None:
    """'binary' si les classes sont {0,1}, 'multi' sinon, None si inconnu."""
    classes = get_model_classes(model)
    if classes is None:
        return None
    return "binary" if set(map(str, classes)) <= {"0", "1"} else "multi"


@st.cache_resource(show_spinner="Entraînement du modèle démo…")
def train_demo_model(_df: pd.DataFrame):
    """Mode démo : RandomForest rapide sur le jeu d'entraînement fourni."""
    X, y = _df[FEATURES], _df.iloc[:, -1]
    return RandomForestClassifier(
        n_estimators=150, n_jobs=-1, random_state=42
    ).fit(X, y)


def split_xy(df: pd.DataFrame):
    """Features dans l'ordre strict + cible (dernière colonne)."""
    return df[FEATURES], df.iloc[:, -1]


def compute_metrics(y_true, y_pred, task: str) -> dict:
    yt = pd.Series(y_true).astype(str)
    yp = pd.Series(y_pred).astype(str)
    if task == "binary":
        kw = dict(average="binary", pos_label="1", zero_division=0)
        f1_label = "F1"
    else:
        kw = dict(average="macro", zero_division=0)
        f1_label = "F1 macro"
    return {
        "Accuracy": accuracy_score(yt, yp),
        "Précision": precision_score(yt, yp, **kw),
        "Rappel": recall_score(yt, yp, **kw),
        f1_label: f1_score(yt, yp, **kw),
    }


def plot_confusion(y_true, y_pred, title: str):
    yt = pd.Series(y_true).astype(str)
    yp = pd.Series(y_pred).astype(str)
    labels = sorted(set(yt) | set(yp))
    cm = confusion_matrix(yt, yp, labels=labels)
    return px.imshow(
        cm,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Prédit", y="Réel", color="Sessions"),
        title=title,
    )


# --------------------------------------------------------------------------
# 2. BARRE LATÉRALE : navigation + gestion des fichiers
# --------------------------------------------------------------------------
st.sidebar.title("🤖 Bot Sessions")
page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Vue d'ensemble",
        "🔍 Exploration",
        "🏆 Performances des modèles",
        "⚡ Prédiction en direct",
    ],
)

with st.sidebar.expander("📁 Fichiers & modèles", expanded=False):
    st.caption(
        "Si un fichier n'est pas trouvé automatiquement, importez-le ici."
    )
    labels = {
        "raw": "bot_sessions.csv",
        "train_m1": "train.csv",
        "test_m1": "test.csv",
        "train_m2": "train (1).csv",
        "test_m2": "test (1).csv",
        "results": "evaluation_results.csv",
    }
    for k, lab in labels.items():
        status = "✅" if find_file(FILES[k]) else "❌"
        st.file_uploader(f"{status} {lab}", type="csv", key=f"upload_{k}")

raw = get_dataset("raw")
train1, test1 = get_dataset("train_m1"), get_dataset("test_m1")
train2, test2 = get_dataset("train_m2"), get_dataset("test_m2")
results = get_dataset("results")

with st.sidebar.expander("🧠 Modèles", expanded=True):
    st.caption(
        "Choisissez un modèle sérialisé (.pkl / .joblib) pour chaque tâche."
    )
    model_files = discover_models()
    options = ["— Aucun / mode démo —"] + [str(p) for p in model_files]

    def pick_model(title: str, key: str, expected: str):
        choice = st.selectbox(title, options, key=f"sel_{key}")
        up = st.file_uploader(
            f"…ou importer un modèle ({title})",
            type=["pkl", "joblib"],
            key=f"up_{key}",
        )
        model, src = None, None
        try:
            if up is not None:
                model, src = load_model_bytes(up.getvalue()), up.name
            elif choice != options[0]:
                model, src = load_model_path(choice), Path(choice).name
        except Exception as e:
            st.error(f"Impossible de charger le modèle : {e}")
        if model is not None:
            t = model_task(model)
            if t is not None and t != expected:
                st.warning(
                    "Ce modèle ne semble pas correspondre à cette tâche "
                    f"(classes : {get_model_classes(model)})."
                )
        return model, src

    model1, src1 = pick_model("Modèle 1 — bot / non-bot", "m1", "binary")
    model2, src2 = pick_model("Modèle 2 — profil de bot", "m2", "multi")
    use_demo = st.checkbox(
        "Autoriser le mode démo (RandomForest entraîné à la volée)",
        value=True,
    )

demo1 = demo2 = False
if (
    model1 is None
    and use_demo
    and train1 is not None
    and set(FEATURES) <= set(train1.columns)
):
    model1, src1, demo1 = train_demo_model(train1), "RandomForest démo", True
if (
    model2 is None
    and use_demo
    and train2 is not None
    and set(FEATURES) <= set(train2.columns)
):
    model2, src2, demo2 = train_demo_model(train2), "RandomForest démo", True


def missing_warning(name: str):
    st.warning(
        f"⚠️ **{name}** introuvable. Placez-le à côté de `app.py` "
        "(ou dans `data/`, `models/`, `results/`) ou importez-le dans la "
        "barre latérale."
    )


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")


# --------------------------------------------------------------------------
# 3. SECTION 1 — VUE D'ENSEMBLE
# --------------------------------------------------------------------------
def page_overview():
    st.title("📊 Vue d'ensemble")
    if raw is None:
        missing_warning("bot_sessions.csv")
    else:
        n = len(raw)
        n_bot = int((raw["label"] == 1).sum()) if "label" in raw else 0
        n_non_bot = n - n_bot
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sessions totales", fmt_int(n))
        c2.metric(
            "Sessions bots (label=1)",
            fmt_int(n_bot),
            f"{n_bot / n:.1%}",
        )
        c3.metric(
            "Sessions non-bots (label=0)",
            fmt_int(n_non_bot),
            f"{n_non_bot / n:.1%}",
        )
        c4.metric("Colonnes", raw.shape[1])

        col_a, col_b = st.columns(2)
        if "label" in raw:
            d = (
                raw["label"]
                .map({0: "Non-bot", 1: "Bot"})
                .value_counts()
                .reset_index()
            )
            d.columns = ["Classe", "Sessions"]
            col_a.plotly_chart(
                px.pie(
                    d,
                    names="Classe",
                    values="Sessions",
                    hole=0.45,
                    title="Bot vs non-bot",
                ),
                use_container_width=True,
            )
        if "profile" in raw:
            bots = raw[raw["label"] == 1] if "label" in raw else raw
            d = bots["profile"].value_counts().reset_index()
            d.columns = ["Profil", "Sessions"]
            col_b.plotly_chart(
                px.bar(
                    d,
                    x="Profil",
                    y="Sessions",
                    color="Profil",
                    text_auto=True,
                    title="Profils de bots (bot_sessions.csv)",
                ),
                use_container_width=True,
            )

    st.subheader("Découpages train / test")
    cols = st.columns(4)
    splits = [
        ("train.csv (M1)", train1),
        ("test.csv (M1)", test1),
        ("train (1).csv (M2)", train2),
        ("test (1).csv (M2)", test2),
    ]
    for col, (name, df) in zip(cols, splits):
        col.metric(name, "—" if df is None else fmt_int(len(df)))

    c1, c2 = st.columns(2)
    for col, (name, df) in zip(
        [c1, c2], [("Modèle 1", train1), ("Modèle 2", train2)]
    ):
        if df is not None:
            tgt = df.columns[-1]
            d = df[tgt].value_counts().reset_index()
            d.columns = [tgt, "Sessions"]
            d[tgt] = d[tgt].astype(str)
            col.plotly_chart(
                px.bar(
                    d,
                    x=tgt,
                    y="Sessions",
                    text_auto=True,
                    color=tgt,
                    title=f"{name} — cible « {tgt} » (train)",
                ),
                use_container_width=True,
            )

    st.subheader("Métriques globales (evaluation_results.csv)")
    if results is None:
        st.info(
            "Aucun fichier de résultats : lancez `evaluation.py` ou "
            "importez-le dans la barre latérale. Les métriques calculées en "
            "direct sont disponibles dans « Performances des modèles »."
        )
    else:
        st.dataframe(results, use_container_width=True)


# --------------------------------------------------------------------------
# 4. SECTION 2 — EXPLORATION
# --------------------------------------------------------------------------
def page_explore():
    st.title("🔍 Exploration visuelle")
    datasets = {
        "bot_sessions.csv (brut)": raw,
        "train.csv (M1)": train1,
        "test.csv (M1)": test1,
        "train (1).csv (M2)": train2,
        "test (1).csv (M2)": test2,
    }
    available = {k: v for k, v in datasets.items() if v is not None}
    if not available:
        missing_warning("aucun jeu de données")
        return

    c1, c2 = st.columns([2, 3])
    name = c1.selectbox("Jeu de données", list(available))
    df = available[name].copy()

    filt_col = next(
        (c for c in ("label", "profile") if c in df.columns), df.columns[-1]
    )
    if filt_col in df.columns and df[filt_col].nunique() <= 10:
        vals = sorted(df[filt_col].astype(str).unique())
        sel = c2.multiselect(f"Filtrer par « {filt_col} »", vals, default=vals)
        df = df[df[filt_col].astype(str).isin(sel)]
    if "profile" in df.columns and filt_col != "profile":
        vals = sorted(df["profile"].dropna().astype(str).unique())
        sel_p = st.multiselect("Filtrer par profil", vals, default=vals)
        df = df[df["profile"].astype(str).isin(sel_p) | df["profile"].isna()]

    st.caption(f"{fmt_int(len(df))} lignes après filtrage")
    if df.empty:
        st.warning("Aucune ligne ne correspond aux filtres.")
        return

    t_ov, t_dist, t_cmp, t_corr = st.tabs([
        "Aperçu & stats",
        "Distributions",
        "Comparaison par classe",
        "Corrélations",
    ])
    num_cols = (
        [c for c in FEATURES if c in df.columns]
        or df.select_dtypes("number").columns.tolist()
    )
    target = filt_col if filt_col in df.columns else None

    with t_ov:
        st.dataframe(df.head(200), use_container_width=True)
        st.subheader("Statistiques descriptives")
        st.dataframe(df[num_cols].describe().T, use_container_width=True)

    with t_dist:
        feat = st.selectbox("Variable", num_cols)
        bins = st.slider("Nombre de classes (bins)", 10, 100, 40)
        color = target if target and df[target].nunique() <= 10 else None
        plot_df = df.assign(**{color: df[color].astype(str)}) if color else df
        st.plotly_chart(
            px.histogram(
                plot_df,
                x=feat,
                color=color,
                nbins=bins,
                barmode="overlay",
                opacity=0.7,
                title=f"Distribution de {feat}",
            ),
            use_container_width=True,
        )

    with t_cmp:
        if target is None or df[target].nunique() > 10:
            st.info("Pas de colonne cible exploitable pour ce jeu de données.")
        else:
            feat2 = st.selectbox(
                "Variable à comparer", num_cols, key="box_feat"
            )
            st.plotly_chart(
                px.box(
                    df.assign(**{target: df[target].astype(str)}),
                    x=target,
                    y=feat2,
                    color=target,
                    title=f"{feat2} par {target}",
                ),
                use_container_width=True,
            )
            grouped = df.groupby(df[target].astype(str))[num_cols]
            means = grouped.mean().round(3)
            st.caption("Moyennes par classe")
            st.dataframe(means, use_container_width=True)

    with t_corr:
        corr = df[num_cols].corr()
        st.plotly_chart(
            px.imshow(
                corr,
                text_auto=".2f",
                zmin=-1,
                zmax=1,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Matrice de corrélation",
            ),
            use_container_width=True,
        )


# --------------------------------------------------------------------------
# 5. SECTION 3 — PERFORMANCES
# --------------------------------------------------------------------------
def evaluate_block(title: str, model, src, demo: bool, test_df, task: str):
    st.subheader(title)
    if model is None:
        st.info(
            "Aucun modèle chargé (et mode démo indisponible). Sélectionnez un "
            "modèle dans la barre latérale."
        )
        return None
    demo_note = " — ⚠️ *mode démo, non issu de votre pipeline*" if demo else ""
    st.caption(f"Modèle : **{src}**{demo_note}")
    if test_df is None:
        missing_warning("le jeu de test correspondant")
        return None
    if not set(FEATURES) <= set(test_df.columns):
        st.error("Le jeu de test ne contient pas les 11 variables attendues.")
        return None
    X, y = split_xy(test_df)
    try:
        y_pred = model.predict(X)
    except Exception as e:
        st.error(
            f"Échec de .predict() : {e}. Vérifiez que le modèle attend bien "
            "ces 11 colonnes (prétraitement inclus dans un Pipeline ou non ?)."
        )
        return None

    m = compute_metrics(y, y_pred, task)
    cols = st.columns(len(m))
    for c, (k, v) in zip(cols, m.items()):
        c.metric(k, f"{v:.4f}")
    st.plotly_chart(
        plot_confusion(y, y_pred, "Matrice de confusion (jeu de test)"),
        use_container_width=True,
    )
    with st.expander("Voir les prédictions"):
        out = X.copy()
        out["réel"], out["prédit"] = y.values, y_pred
        out["correct"] = out["réel"].astype(str) == out["prédit"].astype(str)
        st.dataframe(out.head(500), use_container_width=True)
    return m


def page_performance():
    st.title("🏆 Performances & comparaison des modèles")

    st.subheader("Résultats sauvegardés (evaluation_results.csv)")
    if results is None:
        st.info(
            "Fichier de résultats absent — les métriques ci-dessous sont "
            "recalculées sur les jeux de test."
        )
    else:
        st.dataframe(results, use_container_width=True)
        num = results.select_dtypes("number").columns.tolist()
        txt = results.select_dtypes(exclude="number").columns.tolist()
        if num and txt:
            long = results.melt(
                id_vars=txt[0],
                value_vars=num,
                var_name="Métrique",
                value_name="Score",
            )
            st.plotly_chart(
                px.bar(
                    long,
                    x=txt[0],
                    y="Score",
                    color="Métrique",
                    barmode="group",
                    title="Comparaison des modèles",
                ),
                use_container_width=True,
            )

    t1, t2 = st.tabs(["Modèle 1 — bot / non-bot", "Modèle 2 — profil de bot"])
    with t1:
        evaluate_block(
            "Évaluation sur test.csv", model1, src1, demo1, test1, "binary"
        )
    with t2:
        evaluate_block(
            "Évaluation sur test (1).csv", model2, src2, demo2, test2, "multi"
        )


# --------------------------------------------------------------------------
# 6. SECTION 4 — PRÉDICTION EN DIRECT
# --------------------------------------------------------------------------
def init_inputs():
    for f in FEATURES:
        if f"in_{f}" not in st.session_state:
            st.session_state[f"in_{f}"] = RANGES[f][2]


def apply_preset(name: str):
    for f, v in PRESETS[name].items():
        st.session_state[f"in_{f}"] = int(v) if f in INT_FEATURES else float(v)


def page_predict():
    st.title("⚡ Prédiction interactive")
    if model1 is None and model2 is None:
        st.warning(
            "Aucun modèle disponible. Chargez un modèle ou activez le mode "
            "démo (nécessite train*.csv)."
        )
        return
    if demo1 or demo2:
        st.info(
            "Mode démo : les modèles utilisés sont des RandomForest entraînés "
            "à la volée sur vos CSV."
        )

    init_inputs()
    st.markdown("**Presets rapides**")
    pc = st.columns(len(PRESETS))
    for col, name in zip(pc, PRESETS):
        col.button(
            name, on_click=apply_preset, args=(name,), use_container_width=True
        )

    st.markdown("---")
    left, right = st.columns(2)
    values = {}
    for i, f in enumerate(FEATURES):
        lo, hi, _ = RANGES[f]
        target = left if i % 2 == 0 else right
        with target:
            if f in INT_FEATURES:
                values[f] = st.number_input(
                    f, int(lo), int(hi), step=1, key=f"in_{f}"
                )
            else:
                values[f] = st.number_input(
                    f,
                    float(lo),
                    float(hi),
                    step=0.01,
                    format="%.4f",
                    key=f"in_{f}",
                )

    X = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    with st.expander("DataFrame envoyé au modèle"):
        st.dataframe(X)

    if st.button("🚀 Prédire", type="primary"):
        c1, c2 = st.columns(2)
        try:
            if model1 is not None:
                with c1:
                    st.subheader("Modèle 1 — bot ?")
                    pred = model1.predict(X)[0]
                    is_bot = str(pred) == "1"
                    banner = st.error if is_bot else st.success
                    banner("🤖 **BOT**" if is_bot else "🧑 **NON-BOT**")
                    if hasattr(model1, "predict_proba"):
                        cls1 = [str(c) for c in get_model_classes(model1)]
                        proba = pd.Series(
                            model1.predict_proba(X)[0], index=cls1
                        )
                        p_bot = float(proba.get("1", 0.0))
                        st.metric("Probabilité bot", f"{p_bot:.1%}")
                        st.progress(p_bot)
            if model2 is not None:
                with c2:
                    st.subheader("Modèle 2 — profil du bot")
                    pred2 = model2.predict(X)[0]
                    st.info(f"Profil prédit : **{pred2}**")
                    if hasattr(model2, "predict_proba"):
                        cls2 = [str(c) for c in get_model_classes(model2)]
                        p = pd.DataFrame(
                            {
                                "Profil": cls2,
                                "Probabilité": model2.predict_proba(X)[0],
                            }
                        )
                        st.plotly_chart(
                            px.bar(
                                p,
                                x="Profil",
                                y="Probabilité",
                                color="Profil",
                                range_y=[0, 1],
                            ),
                            use_container_width=True,
                        )
                    st.caption(
                        "Le Modèle 2 n'est pertinent que si la session est "
                        "un bot."
                    )
        except Exception as e:
            st.error(f"Erreur de prédiction : {e}")
            st.caption(
                "Vérifiez que le modèle attend exactement ces 11 colonnes ; "
                "si le modèle n'est pas un Pipeline, le prétraitement "
                "(types, path_entropy…) doit être appliqué en amont."
            )


# --------------------------------------------------------------------------
# 7. ROUTAGE
# --------------------------------------------------------------------------
{
    "📊 Vue d'ensemble": page_overview,
    "🔍 Exploration": page_explore,
    "🏆 Performances des modèles": page_performance,
    "⚡ Prédiction en direct": page_predict,
}[page]()
