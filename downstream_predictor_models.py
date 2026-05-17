"""
AVM pipeline for property price prediction and explainability.

This script loads preprocessed property data, trains several candidate regression models,
computes performance metrics in log price space, and saves diagnostics and reports.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

from meta_categories import TOTAL_FEATURES


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def ensure_dirs() -> None:
    Path("figures").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)


def read_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    df = df[df[target_col].notna()].copy()
    y_raw = df[target_col].copy()
    y_log = np.log1p(np.maximum(y_raw, 0))
    X = df.drop(columns=[target_col])
    return X, y_log, y_raw


def detect_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    available_features = [col for col in TOTAL_FEATURES if col in X.columns]
    if not available_features:
        LOGGER.warning("No TOTAL_FEATURES columns found in input data; falling back to all available columns.")
        feature_frame = X
    else:
        missing = [col for col in TOTAL_FEATURES if col not in X.columns]
        if missing:
            LOGGER.info("Missing expected TOTAL_FEATURES columns: %s", missing)
        feature_frame = X[available_features]

    numeric_cols = feature_frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = feature_frame.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str], for_linear: bool = False) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if for_linear:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_cols),
            ("cat", Pipeline(categorical_steps), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def adjusted_r2(r2: float, n: int, p: int) -> float:
    if n <= p + 1:
        return np.nan
    return 1 - (1 - r2) * ((n - 1) / (n - p - 1))


def evaluate_model(name: str, model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    yhat_train = model.predict(X_train)
    yhat_test = model.predict(X_test)

    try:
        p = model.named_steps["prep"].transform(X_train[:5]).shape[1]
    except Exception:
        p = X_train.shape[1]

    n_train = len(y_train)
    n_test = len(y_test)

    r2_train = r2_score(y_train, yhat_train)
    r2_test = r2_score(y_test, yhat_test)

    metrics = {
        "model": name,
        "mse_log_train": mean_squared_error(y_train, yhat_train),
        "rmse_log_train": rmse(y_train, yhat_train),
        "mae_log_train": mean_absolute_error(y_train, yhat_train),
        "r2_log_train": r2_train,
        "adj_r2_log_train": adjusted_r2(r2_train, n_train, p),
        "mse_log_test": mean_squared_error(y_test, yhat_test),
        "rmse_log_test": rmse(y_test, yhat_test),
        "mae_log_test": mean_absolute_error(y_test, yhat_test),
        "r2_log_test": r2_test,
        "adj_r2_log_test": adjusted_r2(r2_test, n_test, p),
    }
    return metrics, yhat_train, yhat_test


def inverse_log_predictions(y_log_pred: np.ndarray) -> np.ndarray:
    return np.expm1(y_log_pred)


def plot_residuals(y_true_log: np.ndarray, y_pred_log: np.ndarray, title_prefix: str, fname_prefix: str) -> None:
    residuals = y_true_log - y_pred_log

    plt.figure()
    plt.scatter(y_pred_log, residuals, s=8, alpha=0.6)
    plt.axhline(0, linestyle="--", color="gray")
    plt.xlabel("Predicted log(price)")
    plt.ylabel("Residual (log-space)")
    plt.title(f"{title_prefix} — Residuals vs Predicted (log)")
    plt.tight_layout()
    plt.savefig(f"figures/{fname_prefix}_residuals_vs_pred_log.png", dpi=180)
    plt.close()

    plt.figure()
    plt.hist(residuals, bins=40, color="#3b79c4", alpha=0.7)
    plt.title(f"{title_prefix} — Residuals Histogram (log)")
    plt.xlabel("Residual (log-space)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"figures/{fname_prefix}_residuals_hist_log.png", dpi=180)
    plt.close()


def plot_pred_vs_true(y_true_log: np.ndarray, y_pred_log: np.ndarray, title_prefix: str, fname_prefix: str) -> None:
    plt.figure()
    plt.scatter(y_true_log, y_pred_log, s=8, alpha=0.6)
    minv, maxv = np.percentile(np.concatenate([y_true_log, y_pred_log]), [1, 99])
    plt.plot([minv, maxv], [minv, maxv], "--", color="gray")
    plt.xlabel("True log(price)")
    plt.ylabel("Predicted log(price)")
    plt.title(f"{title_prefix} — Predicted vs True (log)")
    plt.tight_layout()
    plt.savefig(f"figures/{fname_prefix}_pred_vs_true_log.png", dpi=180)
    plt.close()


def safe_get_feature_names(preprocessor: ColumnTransformer, X_sample: pd.DataFrame) -> List[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        transformed = preprocessor.transform(X_sample.iloc[:1])
        return [f"feature_{i}" for i in range(transformed.shape[1])]


def fit_with_search(name: str, base_pipeline: Pipeline, param_distributions: Dict[str, List],
                    X_train: pd.DataFrame, y_train: pd.Series,
                    n_iter: int = 25, cv: int = 5, random_state: int = 42) -> Tuple[Pipeline, Dict]:
    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        random_state=random_state,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    LOGGER.info("[%s] Best params: %s", name, search.best_params_)
    return search.best_estimator_, search.best_params_


def map_encoded_to_raw_features(encoded_features: List[str], raw_features: List[str]) -> List[str]:
    raw_sorted = sorted(raw_features, key=len, reverse=True)
    raw_mapping = {
        encoded: next((raw for raw in raw_sorted if encoded.startswith(raw)), encoded)
        for encoded in encoded_features
    }
    mapped = []
    for encoded in encoded_features:
        raw = raw_mapping.get(encoded, encoded)
        if raw not in mapped:
            mapped.append(raw)
    return mapped


def select_pdp_features(best_model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, top_k: int = 6) -> List[str]:
    preprocessor = best_model.named_steps["prep"]
    X_train_enc = preprocessor.transform(X_train)
    encoded_names = safe_get_feature_names(preprocessor, X_train)
    model_obj = best_model.named_steps["model"]

    if hasattr(model_obj, "feature_importances_"):
        importances = model_obj.feature_importances_
        order = np.argsort(importances)[::-1][:top_k * 3]
    else:
        mi = mutual_info_regression(X_train_enc, y_train)
        order = np.argsort(mi)[::-1][:top_k * 3]

    encoded_top = [encoded_names[i] for i in order if i < len(encoded_names)]
    raw_candidates = map_encoded_to_raw_features(encoded_top, list(X_train.columns))
    return raw_candidates[:top_k]


def try_shap(model: Pipeline, X_train_pre: np.ndarray, model_name: str, max_points: int = 5000) -> None:
    if not SHAP_AVAILABLE:
        LOGGER.warning("SHAP is not installed; skipping SHAP plots.")
        return

    try:
        preprocessor = model.named_steps.get("prep")
        if preprocessor is not None:
            try:
                feature_names = list(preprocessor.get_feature_names_out())
            except Exception:
                feature_names = [f"feature_{i}" for i in range(X_train_pre.shape[1])]
        else:
            feature_names = [f"feature_{i}" for i in range(X_train_pre.shape[1])]

        n = X_train_pre.shape[0]
        idx = np.arange(n)
        if n > max_points:
            idx = np.random.choice(n, size=max_points, replace=False)
        background = X_train_pre[idx][:200]

        if any(tag in model_name.lower() for tag in ["forest", "boost", "hist"]):
            explainer = shap.TreeExplainer(model.named_steps["model"])
        else:
            explainer = shap.Explainer(model.predict, background)

        shap_values = explainer(background)
        shap_values.feature_names = feature_names

        plt.figure()
        shap.plots.beeswarm(shap_values, show=False)
        plt.title(f"SHAP Beeswarm — {model_name}")
        plt.tight_layout()
        plt.savefig(f"figures/{model_name}_shap_beeswarm.png", dpi=180)
        plt.close()

        plt.figure()
        shap.plots.bar(shap_values, show=False)
        plt.title(f"SHAP Bar — {model_name}")
        plt.tight_layout()
        plt.savefig(f"figures/{model_name}_shap_bar.png", dpi=180)
        plt.close()

        LOGGER.info("SHAP plots generated for %s", model_name)
    except Exception as exc:
        LOGGER.warning("Could not compute SHAP for %s: %s", model_name, exc)


def run_pca_feature_contribution(best_model: Pipeline, X: pd.DataFrame, y: pd.Series,
                                 output_prefix: str = "pca_feature_contribution",
                                 n_components: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    LOGGER.info("Running PCA contribution analysis using %s components", n_components)
    preprocessor = best_model.named_steps["prep"]
    X_encoded = preprocessor.transform(X)
    feature_names = list(preprocessor.get_feature_names_out())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_scaled)

    var_ratio = pca.explained_variance_ratio_
    cum_var_ratio = np.cumsum(var_ratio)

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(cum_var_ratio) + 1), cum_var_ratio, marker="o")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA — Cumulative Explained Variance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/{output_prefix}_variance.png", dpi=180)
    plt.close()

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i + 1}" for i in range(n_components)],
        index=feature_names,
    )
    contrib = loadings.pow(2)
    contrib_weighted = contrib.mul(var_ratio, axis=1)
    contrib_weighted["total_contribution"] = contrib_weighted.sum(axis=1)

    contrib_sorted = contrib_weighted.sort_values("total_contribution", ascending=False)
    top_features = contrib_sorted.head(20)

    plt.figure(figsize=(7, 5))
    plt.barh(top_features.index[::-1], top_features["total_contribution"][::-1], color="#3b79c4")
    plt.xlabel("Total Variance Contribution (weighted)")
    plt.title("Top Features by PCA Contribution")
    plt.tight_layout()
    plt.savefig(f"figures/{output_prefix}_top_features.png", dpi=180)
    plt.close()

    summary = pd.DataFrame({
        "ExplainedVariance": var_ratio,
        "CumulativeVariance": cum_var_ratio,
    }, index=[f"PC{i + 1}" for i in range(n_components)])
    summary.to_csv(f"reports/{output_prefix}_explained_variance.csv")
    contrib_sorted.to_csv(f"reports/{output_prefix}_feature_contributions.csv")

    LOGGER.info("PCA contribution data saved.")
    return contrib_sorted, summary


def plot_pred_vs_true_distribution(y_true: np.ndarray, y_pred: np.ndarray,
                                   model_name: str = "Model",
                                   output_prefix: str = "pred_vs_true_dist") -> None:
    LOGGER.info("Plotting prediction distributions for %s", model_name)

    df_plot = pd.DataFrame({
        "True Values (log)": y_true,
        "Predicted Values (log)": y_pred,
    })

    plt.figure(figsize=(7, 5))
    sns.kdeplot(df_plot["True Values (log)"], fill=True, label="True (log)", alpha=0.4)
    sns.kdeplot(df_plot["Predicted Values (log)"], fill=True, label="Predicted (log)", alpha=0.4)
    plt.title(f"{model_name} — Distribution (log scale)")
    plt.xlabel("log(price)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{output_prefix}_log.png", dpi=180)
    plt.close()

    df_plot2 = pd.DataFrame({
        "True Values": np.expm1(y_true),
        "Predicted Values": np.expm1(y_pred),
    })

    plt.figure(figsize=(7, 5))
    sns.kdeplot(df_plot2["True Values"], fill=True, label="True", alpha=0.4)
    sns.kdeplot(df_plot2["Predicted Values"], fill=True, label="Predicted", alpha=0.4)
    plt.title(f"{model_name} — Distribution (original scale)")
    plt.xlabel("Price")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{output_prefix}_original.png", dpi=180)
    plt.close()


def main() -> None:
    ensure_dirs()

    parser = argparse.ArgumentParser(description="Train and evaluate AVM models for property price prediction.")
    parser.add_argument("--csv", required=False,
                        default="sample_data/gold_coast.csv",
                        help="Path to the input CSV file.")
    parser.add_argument("--target", default="LISTING_PRICE", help="Target column name.")
    parser.add_argument("--test_size", type=float, default=0.3, help="Test split ratio.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    df = read_data(args.csv)
    LOGGER.info("Loaded dataset with %d rows from %s", len(df), args.csv)
    df = df.dropna(subset=["DESCRIPTION"]).drop_duplicates(subset=["LISTING_ID"]).reset_index(drop=True)
    LOGGER.info("After cleaning, dataset has %d rows", len(df))

    df.head(20).to_csv("reports/preview_head.csv", index=False)
    df.describe(include="all").T.to_csv("reports/describe.csv")

    X, y_log, y_raw = split_features_target(df, args.target)
    numeric_cols, categorical_cols = detect_column_types(X)
    LOGGER.info("Using %d numeric columns and %d categorical columns", len(numeric_cols), len(categorical_cols))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=args.test_size, random_state=args.random_state,
    )
    LOGGER.info("Train/test split: %s / %s", X_train.shape, X_test.shape)

    pre_linear = build_preprocessor(numeric_cols, categorical_cols, for_linear=True)
    pre_tree = build_preprocessor(numeric_cols, categorical_cols, for_linear=False)

    models = {
        "LinearRegression": Pipeline([("prep", pre_linear), ("model", LinearRegression())]),
        "Lasso": Pipeline([("prep", pre_linear), ("model", Lasso(max_iter=5000))]),
        "RandomForest": Pipeline([("prep", pre_tree), ("model", RandomForestRegressor(n_estimators=400, random_state=args.random_state, n_jobs=-1))]),
        "GradientBoosting": Pipeline([("prep", pre_tree), ("model", GradientBoostingRegressor(random_state=args.random_state))]),
        "HistGradientBoosting": Pipeline([("prep", pre_tree), ("model", HistGradientBoostingRegressor(random_state=args.random_state))]),
    }

    tuned_models = {}
    tuned_models["RandomForest"], _ = fit_with_search(
        "RandomForest", models["RandomForest"],
        {
            "model__n_estimators": [300, 400, 600, 800],
            "model__max_depth": [None, 8, 12, 16, 24],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": ["sqrt", "log2", None],
        },
        X_train, y_train, n_iter=20, cv=5, random_state=args.random_state,
    )

    tuned_models["GradientBoosting"], _ = fit_with_search(
        "GradientBoosting", models["GradientBoosting"],
        {
            "model__n_estimators": [200, 300, 500],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.7, 0.9, 1.0],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        X_train, y_train, n_iter=20, cv=5, random_state=args.random_state,
    )

    tuned_models["HistGradientBoosting"], _ = fit_with_search(
        "HistGradientBoosting", models["HistGradientBoosting"],
        {
            "model__max_depth": [None, 6, 8, 12],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_bins": [128, 255],
            "model__l2_regularization": [0.0, 0.001, 0.01],
        },
        X_train, y_train, n_iter=16, cv=5, random_state=args.random_state,
    )

    models["LinearRegression"].fit(X_train, y_train)
    models["Lasso"].fit(X_train, y_train)

    final_models = {
        "LinearRegression": models["LinearRegression"],
        "Lasso": models["Lasso"],
        "RandomForest": tuned_models["RandomForest"],
        "GradientBoosting": tuned_models["GradientBoosting"],
        "HistGradientBoosting": tuned_models["HistGradientBoosting"],
    }

    metrics_rows = []
    for name, model in final_models.items():
        metrics, yhat_train, yhat_test = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        metrics_rows.append(metrics)

        plot_pred_vs_true(y_train.to_numpy(), yhat_train, f"{name} (Train)", f"{name}_train")
        plot_pred_vs_true(y_test.to_numpy(), yhat_test, f"{name} (Test)", f"{name}_test")
        plot_residuals(y_train.to_numpy(), yhat_train, f"{name} (Train)", f"{name}_train")
        plot_residuals(y_test.to_numpy(), yhat_test, f"{name} (Test)", f"{name}_test")

    metrics_df = pd.DataFrame(metrics_rows).sort_values("rmse_log_test")
    metrics_df.to_csv("reports/metrics_summary.csv", index=False)
    LOGGER.info("Saved metrics summary to reports/metrics_summary.csv")
    LOGGER.info("Best models by RMSE:\n%s", metrics_df.to_string(index=False))

    best_name = metrics_df.iloc[0]["model"]
    best_model = final_models[best_name]

    with open("reports/best_model_report.txt", "w") as report_file:
        report_file.write(f"Best model: {best_name}\n\n")
        report_file.write(metrics_df.to_string(index=False))

    preprocessor = best_model.named_steps["prep"]
    X_train_enc = preprocessor.transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    pdp_features = select_pdp_features(best_model, X_train, y_train, top_k=6)

    for feature in pdp_features:
        try:
            PartialDependenceDisplay.from_estimator(best_model, X_test, features=[feature], kind="average")
            plt.title(f"PDP — {best_name} — {feature}")
            plt.tight_layout()
            safe_name = str(feature).replace("/", "_")
            plt.savefig(f"figures/{best_name}_PDP_{safe_name}.png", dpi=180)
            plt.close()
        except Exception as exc:
            LOGGER.warning("Could not plot PDP for %s: %s", feature, exc)

    try_shap(best_model, X_train_enc, best_name)

    y_pred_test_log = best_model.predict(X_test)
    y_pred_test = inverse_log_predictions(y_pred_test_log)
    y_true_test = inverse_log_predictions(y_test.to_numpy())

    predictions_df = pd.DataFrame({
        "y_true_price": y_true_test,
        "y_pred_price": y_pred_test,
        "y_true_log": y_test.reset_index(drop=True),
        "y_pred_log": y_pred_test_log,
    })
    predictions_df["residual_log"] = predictions_df["y_true_log"] - predictions_df["y_pred_log"]
    predictions_df["abs_pct_error_price"] = np.where(
        predictions_df["y_true_price"] > 0,
        np.abs(predictions_df["y_true_price"] - predictions_df["y_pred_price"]) / predictions_df["y_true_price"],
        np.nan,
    )
    predictions_df.to_csv("reports/best_model_predictions_test.csv", index=False)

    LOGGER.info("Saved best model predictions to reports/best_model_predictions_test.csv")
    LOGGER.info("Best model: %s", best_name)

    try:
        plot_pred_vs_true_distribution(y_test.to_numpy(), y_pred_test_log, model_name=best_name,
                                       output_prefix=f"{best_name}_distribution")
    except Exception as exc:
        LOGGER.warning("Distribution visualization failed: %s", exc)

    try:
        run_pca_feature_contribution(best_model, X_test, y_test,
                                     output_prefix="pca_contribution", n_components=10)
    except Exception as exc:
        LOGGER.warning("PCA contribution analysis failed: %s", exc)


if __name__ == "__main__":
    main()
