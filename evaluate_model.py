import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. SETTINGS
# ============================================================

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

TEST_DIR = "data/plant_disease/test"
MODEL_PATH = "plant_disease_cnn.keras"


# ============================================================
# 2. LOAD TEST DATASET
# ============================================================

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

# IMPORTANT:
# Get class names directly from the dataset
class_names = test_ds.class_names

print("\n====================================")
print("CLASSES")
print("====================================")

for i, class_name in enumerate(class_names):
    print(f"{i}: {class_name}")


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

print("\n====================================")
print("LOADING MODEL")
print("====================================")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# 4. FINAL TEST EVALUATION
# ============================================================

print("\n====================================")
print("FINAL TEST EVALUATION")
print("====================================")

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# ============================================================
# 5. GET TRUE LABELS AND PREDICTIONS
# ============================================================

print("\n====================================")
print("GENERATING PREDICTIONS")
print("====================================")

y_true = []
y_pred = []

for images, labels in test_ds:

    # Model outputs probability for each of 13 classes
    predictions = model.predict(
        images,
        verbose=0
    )

    # Select class having highest probability
    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)


# Convert lists to NumPy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)


print(f"Total test images : {len(y_true)}")
print(f"Total predictions : {len(y_pred)}")


# ============================================================
# 6. CLASSIFICATION REPORT
# ============================================================

print("\n====================================")
print("CLASSIFICATION REPORT")
print("====================================")

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(report)


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

print("\n====================================")
print("CONFUSION MATRIX")
print("====================================")

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(len(class_names))
)

print(cm)


# ============================================================
# 8. PLOT CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(16, 14))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Plant Disease CNN - Confusion Matrix")

plt.xticks(
    rotation=90,
    ha="center"
)

plt.yticks(
    rotation=0
)

plt.tight_layout()

plt.show()