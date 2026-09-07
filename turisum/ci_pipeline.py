from pathlib import Path
import json
import sys

import sklearn

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model_building"
REPORT_DIR = ROOT / "reports"
for folder in [DATA_DIR, MODEL_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

TARGET = "ProdTaken"
RANDOM_STATE = 42

def clean_tourism_data(frame):
    cleaned = frame.copy()
    cleaned.columns = cleaned.columns.str.strip()
    cleaned = cleaned.drop(
        columns=[c for c in cleaned.columns if c.startswith("Unnamed:")],
        errors="ignore",
    )
    cleaned = cleaned.drop(columns=["CustomerID"], errors="ignore")

    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].str.strip()

    cleaned["Gender"] = cleaned["Gender"].replace({"Fe Male": "Female"})
    cleaned["Occupation"] = cleaned["Occupation"].replace(
        {"Free Lancer": "Freelancer"}
    )
    cleaned["MaritalStatus"] = cleaned["MaritalStatus"].replace(
        {"Unmarried": "Single"}
    )
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    cleaned[TARGET] = pd.to_numeric(cleaned[TARGET], errors="raise").astype(int)
    return cleaned

raw = pd.read_csv(DATA_DIR / "tourism.csv")
clean = clean_tourism_data(raw)

train_df, test_df = train_test_split(
    clean,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=clean[TARGET],
)
train_df.to_csv(DATA_DIR / "train.csv", index=False)
test_df.to_csv(DATA_DIR / "test.csv", index=False)

X_train = train_df.drop(columns=TARGET)
y_train = train_df[TARGET]
X_test = test_df.drop(columns=TARGET)
y_test = test_df[TARGET]

numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X_train.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

preprocessor = ColumnTransformer(
    [
        ("num", SimpleImputer(strategy="median"), numeric_features),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    ),
                ]
            ),
            categorical_features,
        ),
    ],
    remainder="drop",
)

model_specs = {
    "RandomForest": (
        RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=1,
        ),
        {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 12],
            "model__min_samples_leaf": [1, 3],
            "model__max_features": ["sqrt"],
        },
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.10],
            "model__max_depth": [2, 3],
            "model__min_samples_leaf": [1, 5],
        },
    ),
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

searches = {}
rows = []

for name, (estimator, grid) in model_specs.items():
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("model", estimator)]
    )
    search = GridSearchCV(
        pipeline,
        grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    probabilities = search.best_estimator_.predict_proba(X_test)[:, 1]

    rows.append(
        {
            "model": name,
            "cv_average_precision": search.best_score_,
            "roc_auc": roc_auc_score(y_test, probabilities),
            "average_precision": average_precision_score(y_test, probabilities),
            "best_parameters": json.dumps(search.best_params_, sort_keys=True),
        }
    )
    searches[name] = search

comparison = pd.DataFrame(rows).sort_values(
    "cv_average_precision", ascending=False
)
comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

best_name = comparison.iloc[0]["model"]
best_pipeline = searches[best_name].best_estimator_

oof = cross_val_predict(
    best_pipeline,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=1,
)[:, 1]

pr_precision, pr_recall, thresholds = precision_recall_curve(y_train, oof)
f2 = (5 * pr_precision[:-1] * pr_recall[:-1]) / (
    4 * pr_precision[:-1] + pr_recall[:-1] + 1e-12
)
threshold = float(thresholds[int(np.nanargmax(f2))])

test_probability = best_pipeline.predict_proba(X_test)[:, 1]
test_prediction = (test_probability >= threshold).astype(int)

metrics = {
    "accuracy": accuracy_score(y_test, test_prediction),
    "precision": precision_score(y_test, test_prediction, zero_division=0),
    "recall": recall_score(y_test, test_prediction, zero_division=0),
    "f1": f1_score(y_test, test_prediction, zero_division=0),
    "roc_auc": roc_auc_score(y_test, test_probability),
    "average_precision": average_precision_score(y_test, test_probability),
}

joblib.dump(best_pipeline, MODEL_DIR / "best_model.joblib")

metadata = {
    "model_name": best_name,
    "decision_threshold": threshold,
    "target": TARGET,
    "raw_features": X_train.columns.tolist(),
    "holdout_metrics": {k: float(v) for k, v in metrics.items()},
    "random_state": RANDOM_STATE,
    "environment": {
        "python": sys.version.split()[0],
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    },
}
with (MODEL_DIR / "metadata.json").open("w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2)

print(comparison.to_string(index=False))
print(json.dumps(metadata, indent=2))
