import tensorflow as tf
import numpy as np
from pathlib import Path


# ============================================================
# AGRIVISION
# CNN FEATURE EXTRACTION
# ============================================================

# ============================================================
# 1. SETTINGS
# ============================================================

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

TRAIN_DIR = "data/plant_disease/train"
VAL_DIR = "data/plant_disease/validation"
TEST_DIR = "data/plant_disease/test"

MODEL_PATH = "plant_disease_cnn.keras"

OUTPUT_DIR = Path("data/features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASETS
# ============================================================

print("\n====================================")
print("LOADING DATASETS")
print("====================================")


# ---------------- TRAIN ----------------

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)


# ---------------- VALIDATION ----------------

print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)


# ---------------- TEST ----------------

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)


# ============================================================
# 3. VERIFY CLASS ORDER
# ============================================================

print("\n====================================")
print("VERIFYING CLASS ORDER")
print("====================================")

train_classes = train_ds.class_names
val_classes = val_ds.class_names
test_classes = test_ds.class_names


print("\nTraining classes:")
for i, name in enumerate(train_classes):
    print(f"{i}: {name}")


# Make absolutely sure all datasets use the same
# class-to-index mapping.

if train_classes != val_classes:
    raise ValueError(
        "\nERROR: Training and validation class ordering do not match!"
    )

if train_classes != test_classes:
    raise ValueError(
        "\nERROR: Training and test class ordering do not match!"
    )


class_names = train_classes

print("\nClass ordering verified successfully.")

print(f"Number of classes: {len(class_names)}")


# ============================================================
# 4. LOAD TRAINED CNN
# ============================================================

print("\n====================================")
print("LOADING TRAINED CNN")
print("====================================")

model = tf.keras.models.load_model(MODEL_PATH)

print("CNN loaded successfully.")


# ============================================================
# 5. BUILD / INITIALIZE MODEL
# ============================================================

print("\n====================================")
print("INITIALIZING CNN")
print("====================================")

# Some Keras versions load a Sequential model without
# immediately defining model.input.
#
# Calling the model once creates the required input tensors.

dummy_input = tf.zeros(
    shape=(1, IMG_SIZE[0], IMG_SIZE[1], 3),
    dtype=tf.float32
)

_ = model(dummy_input, training=False)

print("CNN initialized successfully.")


# ============================================================
# 6. FIND GLOBAL AVERAGE POOLING LAYER
# ============================================================

print("\n====================================")
print("LOCATING FEATURE LAYER")
print("====================================")

feature_layer = None

for layer in model.layers:

    if isinstance(
        layer,
        tf.keras.layers.GlobalAveragePooling2D
    ):
        feature_layer = layer
        break


if feature_layer is None:

    raise ValueError(
        "\nERROR: GlobalAveragePooling2D layer was not found "
        "inside the trained CNN."
    )


print("Feature layer found:")
print(f"Layer name : {feature_layer.name}")
print(f"Layer type : {feature_layer.__class__.__name__}")


# ============================================================
# 7. CREATE FEATURE EXTRACTOR
# ============================================================

print("\n====================================")
print("CREATING FEATURE EXTRACTOR")
print("====================================")

feature_extractor = tf.keras.Model(
    inputs=model.inputs,
    outputs=feature_layer.output
)


print("Feature extractor created successfully.")

print(
    "Feature output shape:",
    feature_extractor.output.shape
)


# ============================================================
# 8. VERIFY FEATURE DIMENSION
# ============================================================

feature_dimension = feature_extractor.output_shape[-1]

print("\nFeature dimension:", feature_dimension)


if feature_dimension != 128:

    print(
        "\nWARNING:"
        "\nExpected 128 features based on the current CNN architecture."
        f"\nActual feature dimension: {feature_dimension}"
    )

else:

    print("Confirmed: 128 features per image.")


# ============================================================
# 9. FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_features(dataset, dataset_name):

    print("\n------------------------------------")
    print(f"EXTRACTING {dataset_name} FEATURES")
    print("------------------------------------")

    all_features = []
    all_labels = []

    batch_number = 0

    for images, labels in dataset:

        batch_number += 1

        # IMPORTANT:
        # training=False ensures that layers such as Dropout
        # and training-only behavior are disabled.

        batch_features = feature_extractor(
            images,
            training=False
        )

        all_features.append(
            batch_features.numpy()
        )

        all_labels.append(
            labels.numpy()
        )


    # Combine all batches

    features = np.concatenate(
        all_features,
        axis=0
    )

    labels = np.concatenate(
        all_labels,
        axis=0
    )


    print(f"{dataset_name} extraction complete.")

    print(
        f"{dataset_name} features shape : "
        f"{features.shape}"
    )

    print(
        f"{dataset_name} labels shape   : "
        f"{labels.shape}"
    )

    return features, labels


# ============================================================
# 10. EXTRACT TRAIN FEATURES
# ============================================================

X_train, y_train = extract_features(
    train_ds,
    "TRAIN"
)


# ============================================================
# 11. EXTRACT VALIDATION FEATURES
# ============================================================

X_val, y_val = extract_features(
    val_ds,
    "VALIDATION"
)


# ============================================================
# 12. EXTRACT TEST FEATURES
# ============================================================

print("\n====================================")
print("IMPORTANT TEST-SET RULE")
print("====================================")

print(
    "Test features will be extracted and saved,"
    "\nbut they will NOT be used for training."
)

X_test, y_test = extract_features(
    test_ds,
    "TEST"
)


# ============================================================
# 13. BASIC SANITY CHECKS
# ============================================================

print("\n====================================")
print("SANITY CHECKS")
print("====================================")


# Check number of samples

print("\nNumber of samples:")

print(f"Train      : {len(X_train)}")
print(f"Validation : {len(X_val)}")
print(f"Test       : {len(X_test)}")


# Check labels

print("\nNumber of labels:")

print(f"Train      : {len(y_train)}")
print(f"Validation : {len(y_val)}")
print(f"Test       : {len(y_test)}")


# Check feature dimensions

print("\nFeature dimensions:")

print(f"Train      : {X_train.shape[1]}")
print(f"Validation : {X_val.shape[1]}")
print(f"Test       : {X_test.shape[1]}")


# Verify features and labels match

if len(X_train) != len(y_train):
    raise ValueError("Train features and labels count do not match.")

if len(X_val) != len(y_val):
    raise ValueError(
        "Validation features and labels count do not match."
    )

if len(X_test) != len(y_test):
    raise ValueError(
        "Test features and labels count do not match."
    )


# Verify feature dimensions match

if X_train.shape[1] != X_val.shape[1]:
    raise ValueError(
        "Train and validation feature dimensions do not match."
    )

if X_train.shape[1] != X_test.shape[1]:
    raise ValueError(
        "Train and test feature dimensions do not match."
    )


print("\nAll sanity checks passed.")


# ============================================================
# 14. SAVE FEATURES
# ============================================================

print("\n====================================")
print("SAVING FEATURES")
print("====================================")


# Training

np.save(
    OUTPUT_DIR / "X_train.npy",
    X_train
)

np.save(
    OUTPUT_DIR / "y_train.npy",
    y_train
)


# Validation

np.save(
    OUTPUT_DIR / "X_val.npy",
    X_val
)

np.save(
    OUTPUT_DIR / "y_val.npy",
    y_val
)


# Test

np.save(
    OUTPUT_DIR / "X_test.npy",
    X_test
)

np.save(
    OUTPUT_DIR / "y_test.npy",
    y_test
)


# Class names

np.save(
    OUTPUT_DIR / "class_names.npy",
    np.array(class_names)
)


# ============================================================
# 15. VERIFY SAVED FILES
# ============================================================

print("\n====================================")
print("SAVED FILES")
print("====================================")

saved_files = [
    "X_train.npy",
    "y_train.npy",
    "X_val.npy",
    "y_val.npy",
    "X_test.npy",
    "y_test.npy",
    "class_names.npy"
]

for filename in saved_files:

    file_path = OUTPUT_DIR / filename

    if file_path.exists():

        size_mb = file_path.stat().st_size / (
            1024 * 1024
        )

        print(
            f"OK  {filename:<20} "
            f"{size_mb:.2f} MB"
        )

    else:

        print(
            f"ERROR  {filename} was not created!"
        )


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n====================================")
print("FEATURE EXTRACTION COMPLETE")
print("====================================")

print(f"Classes            : {len(class_names)}")
print(f"Training images    : {len(X_train)}")
print(f"Validation images  : {len(X_val)}")
print(f"Test images        : {len(X_test)}")
print(f"Features per image : {X_train.shape[1]}")

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nThe CNN has successfully converted")
print("each leaf image into a numerical feature vector.")

print("\nNEXT STEP:")
print("Train classical ML models using X_train and y_train.")