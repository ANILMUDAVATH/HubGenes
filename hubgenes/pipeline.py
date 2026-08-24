from __future__ import annotations

import math
import warnings
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings(
    "ignore",
    message="'penalty' was deprecated in version 1.8 and will be removed in 1.10.*",
)

NETWORK_FEATURES = ["betweenness", "closeness", "degree", "mcc"]
FEATURES_TO_STANDARDIZE = ["betweenness", "closeness", "degree", "mcc"]

EDGE_SOURCE_ALIASES = frozenset(
    {
        "#node1",
        "node1",
        "protein1",
        "source",
        "from",
        "interactor_a",
        "node_a",
        "gene1",
    }
)
EDGE_TARGET_ALIASES = frozenset(
    {
        "node2",
        "protein2",
        "target",
        "to",
        "interactor_b",
        "node_b",
        "gene2",
    }
)


def _can_stratify(y: pd.Series) -> bool:
    y = y.astype(int)
    if y.nunique() < 2:
        return False
    return bool(y.value_counts().min() >= 2)


def _zscore_1d(a: np.ndarray) -> np.ndarray:
    m, s = float(np.mean(a)), float(np.std(a))
    if s <= 0:
        return np.zeros_like(a, dtype=float)
    return (a - m) / s


def normalize_interaction_table(df: pd.DataFrame) -> pd.DataFrame:
    """Map input columns to node1/node2 and clean the interaction table."""
    if df is None or df.empty:
        raise ValueError("The interaction table is empty.")

    rename: dict[str, str] = {}
    for c in df.columns:
        low = str(c).lower().strip()
        if low in EDGE_SOURCE_ALIASES:
            rename[c] = "node1"
        elif low in EDGE_TARGET_ALIASES:
            rename[c] = "node2"

    out = df.rename(columns=rename)
    missing = [c for c in ("node1", "node2") if c not in out.columns]
    if missing:
        raise ValueError(
            "Missing required interaction columns after mapping: "
            f"{missing}. Expected columns like #node1/node2 or protein1/protein2."
        )

    use = out[["node1", "node2"]].copy()
    use["node1"] = use["node1"].astype(str).str.strip()
    use["node2"] = use["node2"].astype(str).str.strip()
    use = use[(use["node1"] != "") & (use["node2"] != "")]
    use = use[use["node1"] != use["node2"]]

    if use.empty:
        raise ValueError("No valid non-self interactions were found in the uploaded file.")

    edge_pairs = use.apply(lambda row: tuple(sorted((row["node1"], row["node2"]))), axis=1)
    use[["node_a", "node_b"]] = pd.DataFrame(edge_pairs.tolist(), index=use.index)
    use = use[["node_a", "node_b"]].drop_duplicates().reset_index(drop=True)

    if len(use) < 2:
        raise ValueError("Need at least 2 unique interactions to build the network.")

    return use


def compute_mcc(graph: nx.Graph) -> dict[str, float]:
    """Maximal Clique Centrality stored as float to avoid downstream integer overflow in UI/display layers."""
    mcc = {node: 0.0 for node in graph.nodes}
    for clique in nx.find_cliques(graph):
        clique_weight = float(math.factorial(len(clique) - 1))
        for node in clique:
            mcc[node] += clique_weight
    return mcc


def build_topology_features(interactions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    graph = nx.from_pandas_edgelist(interactions, source="node_a", target="node_b")
    if graph.number_of_nodes() < 3:
        raise ValueError("Need at least 3 unique genes in the network.")

    degree_dict = dict(graph.degree())
    betweenness_dict = nx.betweenness_centrality(graph, normalized=True)
    closeness_dict = nx.closeness_centrality(graph)
    mcc_dict = compute_mcc(graph)

    features_df = pd.DataFrame({"name": sorted(graph.nodes())})
    features_df["betweenness"] = features_df["name"].map(betweenness_dict)
    features_df["closeness"] = features_df["name"].map(closeness_dict)
    features_df["degree"] = features_df["name"].map(degree_dict)
    features_df["mcc"] = pd.to_numeric(features_df["name"].map(mcc_dict), errors="coerce").astype(float)

    graph_summary = {
        "nodes": int(graph.number_of_nodes()),
        "edges": int(graph.number_of_edges()),
        "connected_components": int(nx.number_connected_components(graph)),
        "density": float(nx.density(graph)),
    }
    return features_df, graph_summary


def add_median_labels(features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    out = features_df.copy()
    medians = out[NETWORK_FEATURES].median()
    out["label"] = (
        (out["betweenness"] >= medians["betweenness"])
        & (out["closeness"] >= medians["closeness"])
        & (out["degree"] >= medians["degree"])
        & (out["mcc"] >= medians["mcc"])
    ).astype(int)
    return out, medians


def rank_genes_from_interactions(
    df: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Build STRING graph features, label genes by the median rule across four topology measures,
    train LASSO + SVM-RFE + RF, then rank genes by the composite score.

    Returns (ranking_df, topology_features_df, info).
    """
    interactions = normalize_interaction_table(df)
    topology_df, graph_summary = build_topology_features(interactions)
    topology_df, medians = add_median_labels(topology_df)

    X = topology_df[NETWORK_FEATURES].copy()
    y = topology_df["label"].astype(int)
    n = len(topology_df)
    use_stratify = _can_stratify(y) and n >= 10

    if n >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=random_state,
            stratify=y if use_stratify else None,
        )
    else:
        X_train, X_test, y_train, y_test = X.copy(), None, y.copy(), None

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_train[FEATURES_TO_STANDARDIZE] = scaler.fit_transform(X_train[FEATURES_TO_STANDARDIZE])

    X_all = topology_df[NETWORK_FEATURES].copy()
    X_all[FEATURES_TO_STANDARDIZE] = scaler.transform(topology_df[FEATURES_TO_STANDARDIZE])
    X_tr = X_train[NETWORK_FEATURES].values
    y_tr = y_train.values
    X_full = X_all.values

    lasso_lr = LogisticRegression(
        penalty="elasticnet",
        l1_ratio=1.0,
        solver="saga",
        max_iter=10_000,
        random_state=random_state,
    )
    lasso_lr.fit(X_tr, y_tr)
    score_lasso = lasso_lr.decision_function(X_full)

    n_features = min(2, X_tr.shape[1])
    rfe = RFE(
        estimator=LinearSVC(random_state=random_state, dual="auto"),
        n_features_to_select=n_features,
        step=1,
    )
    rfe.fit(X_tr, y_tr)
    score_svm_rfe = rfe.estimator_.decision_function(rfe.transform(X_full))

    rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    rf.fit(X_tr, y_tr)
    score_rf = rf.predict_proba(X_full)[:, 1]

    z_lasso = _zscore_1d(score_lasso)
    z_svm = _zscore_1d(score_svm_rfe)
    z_rf = _zscore_1d(score_rf)
    composite_score = (z_lasso + z_svm + z_rf) / 3.0

    kept = np.array(NETWORK_FEATURES)[rfe.support_].tolist()

    ranking = (
        pd.DataFrame(
            {
                "name": topology_df["name"].values,
                "betweenness": topology_df["betweenness"].values,
                "closeness": topology_df["closeness"].values,
                "degree": topology_df["degree"].values,
                "mcc": topology_df["mcc"].values,
                "label": topology_df["label"].values,
                "score_lasso": score_lasso,
                "score_svm_rfe": score_svm_rfe,
                "score_rf": score_rf,
                "composite_score": composite_score,
            }
        )
        .sort_values("composite_score", ascending=False)
        .reset_index(drop=True)
    )
    ranking["rank"] = ranking.index + 1
    ranking = ranking[
        [
            "rank",
            "name",
            "betweenness",
            "closeness",
            "degree",
            "mcc",
            "label",
            "score_lasso",
            "score_svm_rfe",
            "score_rf",
            "composite_score",
        ]
    ]

    evaluation_rows: list[dict[str, Any]] = []
    if X_test is not None and y_test is not None:
        X_test = X_test.copy()
        X_test[FEATURES_TO_STANDARDIZE] = scaler.transform(X_test[FEATURES_TO_STANDARDIZE])
        evaluation_rows = [
            {
                "model": "LASSO logistic regression",
                "test_accuracy": accuracy_score(y_test, lasso_lr.predict(X_test.values)),
                "test_roc_auc": roc_auc_score(y_test, lasso_lr.decision_function(X_test.values)),
            },
            {
                "model": "SVM-RFE (LinearSVC)",
                "test_accuracy": accuracy_score(y_test, rfe.predict(X_test.values)),
                "test_roc_auc": roc_auc_score(
                    y_test,
                    rfe.estimator_.decision_function(rfe.transform(X_test.values)),
                ),
            },
            {
                "model": "Random Forest",
                "test_accuracy": accuracy_score(y_test, rf.predict(X_test.values)),
                "test_roc_auc": roc_auc_score(y_test, rf.predict_proba(X_test.values)[:, 1]),
            },
        ]

    info = {
        "graph_summary": graph_summary,
        "medians": medians.to_dict(),
        "rfe_features": kept,
        "n_train": int(len(X_train)),
        "n_total": int(n),
        "used_stratify": use_stratify,
        "evaluation": pd.DataFrame(evaluation_rows),
        "interactions": interactions,
    }
    return ranking, topology_df, info
