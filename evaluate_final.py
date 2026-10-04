import os
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. SETTINGS
# ============================================================

FEATURE_DIR = "data/features"
MODEL_DIR = "data/ml_models"

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_model.joblib"
)

TEST_FEATURES_PATH = os.path.join(
    FEATURE_DIR,
    "X_test.npy"
)

TEST_LABELS_PATH = os.path.join(
    FEATURE_DIR,
    "y_test.npy"
)

CLASS_NAMES_PATH = os.path.join(
    FEATURE_DIR,
    "class_names.npy"
)


# ============================================================
# 2. HEADER
# ============================================================

print("\n====================================")
print("FINAL TEST EVALUATION")
print("====================================")

print("\nIMPORTANT:")
print("The test set was NOT used during model selection.")
print("This evaluation is performed only once on the final model.")


# ============================================================
# 3. LOAD TEST FEATURES
# ============================================================

print("\n====================================")
print("LOADING TEST FEATURES")
print("====================================")

X_test = np.load(
    TEST_FEATURES_PATH
)

y_test = np.load(
    TEST_LABELS_PATH
)

class_names = np.load(
    CLASS_NAMES_PATH,
    allow_pickle=True
).tolist()


print("X_test shape :", X_test.shape)
print("y_test shape :", y_test.shape)

print("Number of classes :", len(class_names))


# ============================================================
# 4. SANITY CHECKS
# ============================================================

print("\n====================================")
print("SANITY CHECKS")
print("====================================")

assert X_test.shape[0] == y_test.shape[0], \
    "Test feature/label count mismatch."

assert X_test.shape[1] == 128, \
    "Unexpected feature dimension."

assert len(class_names) == 13, \
    "Unexpected number of classes."

assert np.isfinite(X_test).all(), \
    "X_test contains NaN or infinite values."

assert np.isfinite(y_test).all(), \
    "y_test contains invalid values."


print("All sanity checks passed.")


# ============================================================
# 5. LOAD BEST MODEL
# ============================================================

print("\n====================================")
print("LOADING FINAL MODEL")
print("====================================")

model = joblib.load(
    BEST_MODEL_PATH
)

print("Model loaded successfully.")
print("Model type :", type(model).__name__)


# ============================================================
# 6. GENERATE TEST PREDICTIONS
# ============================================================

print("\n====================================")
print("GENERATING TEST PREDICTIONS")
print("====================================")

y_pred = model.predict(
    X_test
)

print("Predictions generated successfully.")

print("Total test samples :", len(y_test))
print("Total predictions   :", len(y_pred))


# ============================================================
# 7. VERIFY PREDICTIONS
# ============================================================

assert len(y_test) == len(y_pred), \
    "Prediction count does not match test labels."


# ============================================================
# 8. FINAL METRICS
# ============================================================

print("\n====================================")
print("FINAL TEST RESULTS")
print("====================================")

test_accuracy = accuracy_score(
    y_test,
    y_pred
)

test_precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

test_recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)


print(f"Test Accuracy        : {test_accuracy:.4f}")
print(f"Test Accuracy        : {test_accuracy * 100:.2f}%")

print(f"Macro Precision      : {test_precision:.4f}")
print(f"Macro Recall         : {test_recall:.4f}")
print(f"Macro F1             : {test_f1:.4f}")


# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

print("\n====================================")
print("FINAL CLASSIFICATION REPORT")
print("====================================")

report = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(report)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

print("\n====================================")
print("FINAL CONFUSION MATRIX")
print("====================================")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ============================================================
# 11. PER-CLASS SUMMARY
# ============================================================

print("\n====================================")
print("PER-CLASS PERFORMANCE")
print("====================================")

precision_per_class = precision_score(
    y_test,
    y_pred,
    average=None,
    labels=np.arange(len(class_names)),
    zero_division=0
)

recall_per_class = recall_score(
    y_test,
    y_pred,
    average=None,
    labels=np.arange(len(class_names)),
    zero_division=0
)

f1_per_class = f1_score(
    y_test,
    y_pred,
    average=None,
    labels=np.arange(len(class_names)),
    zero_division=0
)

support_per_class = np.bincount(
    y_test,
    minlength=len(class_names)
)


for i, class_name in enumerate(class_names):

    print(
        f"\n{i}: {class_name}"
    )

    print(
        f"   Precision : {precision_per_class[i]:.4f}"
    )

    print(
        f"   Recall    : {recall_per_class[i]:.4f}"
    )

    print(
        f"   F1-score  : {f1_per_class[i]:.4f}"
    )

    print(
        f"   Support   : {support_per_class[i]}"
    )


# ============================================================
# 12. SAVE FINAL METRICS
# ============================================================

print("\n====================================")
print("SAVING FINAL RESULTS")
print("====================================")


final_results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1"
    ],

    "Value": [
        test_accuracy,
        test_precision,
        test_recall,
        test_f1
    ]
})


metrics_path = os.path.join(
    MODEL_DIR,
    "final_test_metrics.csv"
)

final_results.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# 13. SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

cm_path = os.path.join(
    MODEL_DIR,
    "final_confusion_matrix.csv"
)

cm_df.to_csv(
    cm_path
)


# ============================================================
# 14. SAVE PER-CLASS RESULTS
# ============================================================

per_class_df = pd.DataFrame({
    "Class": class_names,
    "Precision": precision_per_class,
    "Recall": recall_per_class,
    "F1_Score": f1_per_class,
    "Support": support_per_class
})


per_class_path = os.path.join(
    MODEL_DIR,
    "final_per_class_results.csv"
)

per_class_df.to_csv(
    per_class_path,
    index=False
)


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n====================================")
print("FINAL EVALUATION COMPLETE")
print("====================================")

print(
    f"Final Test Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Final Macro F1      : "
    f"{test_f1:.4f}"
)

print("\nSaved files:")

print(
    f"OK  {metrics_path}"
)

print(
    f"OK  {cm_path}"
)

print(
    f"OK  {per_class_path}"
)

print("\n====================================")
print("AGriVISION CNN + SVM PIPELINE")
print("====================================")

print(
    "CNN          : Feature extractor"
)

print(
    "Features     : 128-dimensional"
)

print(
    "ML Classifier: SVM"
)

print(
    "Classes      : 13"
)

print(
    "Test Images  : "
    f"{len(y_test)}"
)

print(
    f"Final Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)

print("\nTest set remained completely separate")
print("from model selection.")