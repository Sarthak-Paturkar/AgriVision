import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. SETTINGS
# ============================================================

FEATURE_DIR = "data/features"
MODEL_DIR = "data/ml_models"

os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# 2. LOAD FEATURES
# ============================================================

print("\n====================================")
print("LOADING CNN FEATURES")
print("====================================")

X_train = np.load(
    os.path.join(FEATURE_DIR, "X_train.npy")
)

y_train = np.load(
    os.path.join(FEATURE_DIR, "y_train.npy")
)

X_val = np.load(
    os.path.join(FEATURE_DIR, "X_val.npy")
)

y_val = np.load(
    os.path.join(FEATURE_DIR, "y_val.npy")
)

class_names = np.load(
    os.path.join(FEATURE_DIR, "class_names.npy"),
    allow_pickle=True
).tolist()


print("\nFeature shapes:")

print("X_train :", X_train.shape)
print("y_train :", y_train.shape)

print("X_val   :", X_val.shape)
print("y_val   :", y_val.shape)

print("\nNumber of classes:", len(class_names))


# ============================================================
# 3. SANITY CHECKS
# ============================================================

print("\n====================================")
print("SANITY CHECKS")
print("====================================")

assert X_train.shape[0] == y_train.shape[0], \
    "Training feature/label count mismatch."

assert X_val.shape[0] == y_val.shape[0], \
    "Validation feature/label count mismatch."

assert X_train.shape[1] == X_val.shape[1], \
    "Train and validation feature dimensions mismatch."

assert len(np.unique(y_train)) == len(class_names), \
    "Number of classes in training data does not match class_names."

assert np.isfinite(X_train).all(), \
    "X_train contains NaN or infinite values."

assert np.isfinite(X_val).all(), \
    "X_val contains NaN or infinite values."


print("All sanity checks passed.")


# ============================================================
# 4. DISPLAY CLASS DISTRIBUTION
# ============================================================

print("\n====================================")
print("TRAINING CLASS DISTRIBUTION")
print("====================================")

unique_classes, counts = np.unique(
    y_train,
    return_counts=True
)

for class_id, count in zip(unique_classes, counts):

    print(
        f"{class_id:2d}: "
        f"{class_names[class_id]:50s} "
        f"{count:5d}"
    )


# ============================================================
# 5. MODEL STORAGE
# ============================================================

results = []

trained_models = {}


# ============================================================
# 6. SVM
# ============================================================

print("\n====================================")
print("TRAINING SVM")
print("====================================")

print("Using RBF kernel...")
print("Using StandardScaler...")
print("Using balanced class weights...")

svm_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),

    (
        "classifier",
        SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE
        )
    )
])


svm_model.fit(
    X_train,
    y_train
)


svm_predictions = svm_model.predict(
    X_val
)


svm_accuracy = accuracy_score(
    y_val,
    svm_predictions
)

svm_precision = precision_score(
    y_val,
    svm_predictions,
    average="macro",
    zero_division=0
)

svm_recall = recall_score(
    y_val,
    svm_predictions,
    average="macro",
    zero_division=0
)

svm_f1 = f1_score(
    y_val,
    svm_predictions,
    average="macro",
    zero_division=0
)


print("\nSVM VALIDATION RESULTS")
print("------------------------------------")

print(f"Accuracy        : {svm_accuracy:.4f}")
print(f"Macro Precision : {svm_precision:.4f}")
print(f"Macro Recall    : {svm_recall:.4f}")
print(f"Macro F1        : {svm_f1:.4f}")


results.append({
    "Model": "SVM",
    "Accuracy": svm_accuracy,
    "Macro_Precision": svm_precision,
    "Macro_Recall": svm_recall,
    "Macro_F1": svm_f1
})

trained_models["SVM"] = svm_model


# ============================================================
# 7. RANDOM FOREST
# ============================================================

print("\n====================================")
print("TRAINING RANDOM FOREST")
print("====================================")

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1
)


rf_model.fit(
    X_train,
    y_train
)


rf_predictions = rf_model.predict(
    X_val
)


rf_accuracy = accuracy_score(
    y_val,
    rf_predictions
)

rf_precision = precision_score(
    y_val,
    rf_predictions,
    average="macro",
    zero_division=0
)

rf_recall = recall_score(
    y_val,
    rf_predictions,
    average="macro",
    zero_division=0
)

rf_f1 = f1_score(
    y_val,
    rf_predictions,
    average="macro",
    zero_division=0
)


print("\nRANDOM FOREST VALIDATION RESULTS")
print("------------------------------------")

print(f"Accuracy        : {rf_accuracy:.4f}")
print(f"Macro Precision : {rf_precision:.4f}")
print(f"Macro Recall    : {rf_recall:.4f}")
print(f"Macro F1        : {rf_f1:.4f}")


results.append({
    "Model": "Random Forest",
    "Accuracy": rf_accuracy,
    "Macro_Precision": rf_precision,
    "Macro_Recall": rf_recall,
    "Macro_F1": rf_f1
})

trained_models["Random Forest"] = rf_model


# ============================================================
# 8. LOGISTIC REGRESSION
# ============================================================

print("\n====================================")
print("TRAINING LOGISTIC REGRESSION")
print("====================================")

logistic_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE
        )
    )
])


logistic_model.fit(
    X_train,
    y_train
)


logistic_predictions = logistic_model.predict(
    X_val
)


logistic_accuracy = accuracy_score(
    y_val,
    logistic_predictions
)

logistic_precision = precision_score(
    y_val,
    logistic_predictions,
    average="macro",
    zero_division=0
)

logistic_recall = recall_score(
    y_val,
    logistic_predictions,
    average="macro",
    zero_division=0
)

logistic_f1 = f1_score(
    y_val,
    logistic_predictions,
    average="macro",
    zero_division=0
)


print("\nLOGISTIC REGRESSION VALIDATION RESULTS")
print("------------------------------------")

print(f"Accuracy        : {logistic_accuracy:.4f}")
print(f"Macro Precision : {logistic_precision:.4f}")
print(f"Macro Recall    : {logistic_recall:.4f}")
print(f"Macro F1        : {logistic_f1:.4f}")


results.append({
    "Model": "Logistic Regression",
    "Accuracy": logistic_accuracy,
    "Macro_Precision": logistic_precision,
    "Macro_Recall": logistic_recall,
    "Macro_F1": logistic_f1
})

trained_models["Logistic Regression"] = logistic_model


# ============================================================
# 9. XGBOOST
# ============================================================

print("\n====================================")
print("TRAINING XGBOOST")
print("====================================")

try:

    from xgboost import XGBClassifier

    print("XGBoost library found.")

    xgb_model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        num_class=len(class_names),
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


    xgb_model.fit(
        X_train,
        y_train
    )


    xgb_predictions = xgb_model.predict(
        X_val
    )


    xgb_accuracy = accuracy_score(
        y_val,
        xgb_predictions
    )

    xgb_precision = precision_score(
        y_val,
        xgb_predictions,
        average="macro",
        zero_division=0
    )

    xgb_recall = recall_score(
        y_val,
        xgb_predictions,
        average="macro",
        zero_division=0
    )

    xgb_f1 = f1_score(
        y_val,
        xgb_predictions,
        average="macro",
        zero_division=0
    )


    print("\nXGBOOST VALIDATION RESULTS")
    print("------------------------------------")

    print(f"Accuracy        : {xgb_accuracy:.4f}")
    print(f"Macro Precision : {xgb_precision:.4f}")
    print(f"Macro Recall    : {xgb_recall:.4f}")
    print(f"Macro F1        : {xgb_f1:.4f}")


    results.append({
        "Model": "XGBoost",
        "Accuracy": xgb_accuracy,
        "Macro_Precision": xgb_precision,
        "Macro_Recall": xgb_recall,
        "Macro_F1": xgb_f1
    })

    trained_models["XGBoost"] = xgb_model

except ImportError:

    print("\nWARNING:")
    print("XGBoost is not installed.")
    print("Skipping XGBoost.")
    print(
        "Install it using: pip install xgboost"
    )


# ============================================================
# 10. MODEL COMPARISON
# ============================================================

print("\n====================================")
print("MODEL COMPARISON")
print("====================================")


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro_F1",
    ascending=False
).reset_index(drop=True)


print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Macro_Precision": "{:.4f}".format,
            "Macro_Recall": "{:.4f}".format,
            "Macro_F1": "{:.4f}".format
        }
    )
)


# ============================================================
# 11. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]


print("\n====================================")
print("BEST MODEL")
print("====================================")

print("Selected model :", best_model_name)

print(
    f"Validation Accuracy : "
    f"{results_df.iloc[0]['Accuracy']:.4f}"
)

print(
    f"Validation Macro F1 : "
    f"{results_df.iloc[0]['Macro_F1']:.4f}"
)


# ============================================================
# 12. SAVE ALL MODELS
# ============================================================

print("\n====================================")
print("SAVING MODELS")
print("====================================")


for model_name, trained_model in trained_models.items():

    filename = (
        model_name
        .lower()
        .replace(" ", "_")
        + ".joblib"
    )

    path = os.path.join(
        MODEL_DIR,
        filename
    )

    joblib.dump(
        trained_model,
        path
    )

    print(
        f"Saved: {path}"
    )


# ============================================================
# 13. SAVE BEST MODEL
# ============================================================

best_model_path = os.path.join(
    MODEL_DIR,
    "best_model.joblib"
)


joblib.dump(
    best_model,
    best_model_path
)


print(
    f"\nBest model saved as:\n"
    f"{best_model_path}"
)


# ============================================================
# 14. SAVE MODEL COMPARISON
# ============================================================

comparison_path = os.path.join(
    MODEL_DIR,
    "model_comparison.csv"
)


results_df.to_csv(
    comparison_path,
    index=False
)


print(
    f"Model comparison saved as:\n"
    f"{comparison_path}"
)


# ============================================================
# 15. DETAILED REPORT FOR BEST MODEL
# ============================================================

print("\n====================================")
print("BEST MODEL - DETAILED VALIDATION")
print("====================================")


best_predictions = best_model.predict(
    X_val
)


print(
    classification_report(
        y_val,
        best_predictions,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

print("\n====================================")
print("CONFUSION MATRIX")
print("====================================")


cm = confusion_matrix(
    y_val,
    best_predictions
)


print(cm)


# ============================================================
# 17. FINAL MESSAGE
# ============================================================

print("\n====================================")
print("CLASSICAL ML TRAINING COMPLETE")
print("====================================")

print(
    f"Best model : {best_model_name}"
)

print(
    f"Validation samples : {len(y_val)}"
)

print(
    f"Feature dimension : {X_train.shape[1]}"
)

print("\nIMPORTANT:")
print(
    "The test set was NOT used for model selection."
)

print(
    "The test set should now be used ONLY "
    "for final evaluation."
)

print("\nNext step:")
print(
    "Run the final test evaluation using "
    "the selected best model."
)