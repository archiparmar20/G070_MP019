
import argparse
import hashlib
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .clean import clean_train, MODEL_INPUT_COLUMNS
from .validate_data import validate_training_data, NUMERIC_FEATURES, CATEGORICAL_FEATURES


# --------------------------------------------------
# Project paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "raw" / "train.csv"
PARAMS_PATH = ROOT / "params.yaml"
MLFLOW_DB = ROOT / "mlflow.db"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"

EXPERIMENT_NAME = "mp019-energy-kwh-regression"
REGISTERED_MODEL_NAME = "mp019-energy-predictor"

TARGET = "energy_kwh"
TEST_SIZE = 0.20
RANDOM_STATE = 42


# --------------------------------------------------
# Utility functions
# --------------------------------------------------

def calculate_sha256(file_path):
    """Calculate SHA-256 hash of the original CSV file."""
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def load_config(config_path):
    """Load project configuration from params.yaml."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_preprocessor():
    """Build preprocessing for numeric and categorical features."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0.0,
    )

    return preprocessor


def build_pipeline(model):
    """Combine preprocessing and regression model."""

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", model),
        ]
    )


# --------------------------------------------------
# Training function
# --------------------------------------------------

def train(args):
    print("\n========== MP019 MODEL TRAINING ==========\n")

    # 1. Verify required files
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found: {DATA_PATH}")

    if not PARAMS_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found: {PARAMS_PATH}")

    # 2. Load original dataset and calculate its hash
    print("Loading original training dataset...")

    raw_df = pd.read_csv(DATA_PATH)
    data_sha256 = calculate_sha256(DATA_PATH)

    print(f"Original dataset shape: {raw_df.shape}")
    print(f"Dataset SHA-256: {data_sha256}")

    # 3. Load configuration
    config = load_config(PARAMS_PATH)

    # 4. Clean training data using repository implementation
    print("\nCleaning training data...")

    cleaned_df = clean_train(raw_df, config)

    # 5. Validate cleaned data
    print("\nValidating cleaned training data...")

    validate_training_data(cleaned_df)

    print("Training data validation successful.")

    # 6. Select exactly eight approved features
    expected_features = [
        "battery_start_pct",
        "session_minutes",
        "charger_kw",
        "temp_c",
        "vehicle_age_yr",
        "queue_before",
        "vehicle",
        "station_type",
    ]

    if set(MODEL_INPUT_COLUMNS) != set(expected_features):
        raise ValueError(
            "MODEL_INPUT_COLUMNS does not match the eight approved features."
        )

    X = cleaned_df[expected_features].copy()
    y = pd.to_numeric(cleaned_df[TARGET], errors="raise")

    # Ensure numeric columns are numeric
    for column in NUMERIC_FEATURES:
        X[column] = pd.to_numeric(X[column], errors="raise")

    # Explicitly ensure the leakage column is excluded
    if "kwh_billed_in_app" in X.columns:
        raise ValueError("Leakage column kwh_billed_in_app must not be used.")

    print(f"\nNumber of input features: {len(X.columns)}")
    print(f"Features: {list(X.columns)}")
    print(f"Target: {TARGET}")

    # 7. Create a reproducible 80:20 split
    X_train, X_validation, y_train, y_validation = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"\nTraining rows: {len(X_train)}")
    print(f"Validation rows: {len(X_validation)}")

    # 8. Define at least three model configurations
    model_candidates = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=2,
            loss="squared_error",
            random_state=RANDOM_STATE,
        ),
    }

    # 9. Configure MLflow tracking
    tracking_uri = f"sqlite:///{MLFLOW_DB.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []

    # 10. Train and evaluate each candidate
    print("\n========== MODEL COMPARISON ==========\n")

    for model_name, model in model_candidates.items():

        print(f"Training {model_name}...")

        pipeline = build_pipeline(model)

        with mlflow.start_run(run_name=model_name) as run:

            # Fit model
            pipeline.fit(X_train, y_train)

            # Predict validation data
            predictions = pipeline.predict(X_validation)

            # Calculate validation MAE
            mae = mean_absolute_error(y_validation, predictions)

            # Log model parameters
            mlflow.log_params(model.get_params())

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("training_rows", len(X_train))
            mlflow.log_param("validation_rows", len(X_validation))
            mlflow.log_param("data_sha256", data_sha256)

            # Log validation metric
            mlflow.log_metric("mae_validation", mae)

            # Log fitted model and retain its exact MLflow model URI
            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                ],
            )

            results.append(
                {
                    "model_name": model_name,
                    "run_id": run.info.run_id,
                    "mae_validation": mae,
                    "model_uri": model_info.model_uri,
                    "pipeline": pipeline,
                }
            )

            print(f"{model_name} validation MAE: {mae:.6f}")
            print(f"Run ID: {run.info.run_id}")

    # 11. Select the model with the lowest validation MAE
    winner = min(
        results,
        key=lambda result: result["mae_validation"],
    )

    print("\n========== BEST MODEL ==========\n")
    print(f"Best model: {winner['model_name']}")
    print(f"Best validation MAE: {winner['mae_validation']:.6f}")
    print(f"Best run ID: {winner['run_id']}")

    # 12. Save the winning fitted pipeline locally
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(winner["pipeline"], MODEL_PATH)

    print(f"\nWinning pipeline saved to: {MODEL_PATH}")

    # 13. Register the winning model in MLflow
    print("\nRegistering winning model in MLflow...")

    registered_version = mlflow.register_model(
        model_uri=winner["model_uri"],
        name=args.model_name,
        await_registration_for=300,
    )

    # 14. Assign champion alias
    client = mlflow.tracking.MlflowClient()

    client.set_registered_model_alias(
        name=args.model_name,
        alias="champion",
        version=registered_version.version,
    )

    print("\n========== REGISTRATION DETAILS ==========\n")
    print(f"Experiment name: {EXPERIMENT_NAME}")
    print(f"Registered model: {args.model_name}")
    print(f"Registered version: {registered_version.version}")
    print("Model alias: champion")

    # 15. Print comparison results
    print("\n========== VALIDATION RESULTS ==========\n")

    for result in sorted(
        results,
        key=lambda item: item["mae_validation"],
    ):
        print(
            f"{result['model_name']}: "
            f"MAE = {result['mae_validation']:.6f}"
        )

    print("\n========== TRAINING COMPLETED ==========\n")

    print(f"Best run ID: {winner['run_id']}")
    print(f"Model file: {MODEL_PATH}")
    print(f"MLflow database: {MLFLOW_DB}")


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train and register MP019 energy prediction models."
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=REGISTERED_MODEL_NAME,
        help="Name of the MLflow registered model.",
    )

    args = parser.parse_args()

    train(args)