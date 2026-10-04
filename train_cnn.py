import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np


# ============================================================
# 1. SETTINGS
# ============================================================

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
NUM_CLASSES = 13
EPOCHS = 10


# ============================================================
# 2. DATASET PATHS
# ============================================================

TRAIN_DIR = "data/plant_disease/train"
VAL_DIR = "data/plant_disease/validation"
TEST_DIR = "data/plant_disease/test"


# ============================================================
# 3. LOAD TRAINING DATA
# ============================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=42
)


# ============================================================
# 4. LOAD VALIDATION DATA
# ============================================================

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)


# ============================================================
# 5. LOAD TEST DATA
# ============================================================

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)


# ============================================================
# 6. GET CLASS NAMES
# ============================================================

class_names = train_ds.class_names

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")


# ============================================================
# 7. CALCULATE CLASS WEIGHTS
# ============================================================

# Collect all training labels
y_train = np.concatenate(
    [y.numpy() for x, y in train_ds],
    axis=0
)

# Count how many images belong to each class
class_counts = np.bincount(y_train, minlength=NUM_CLASSES)

total_samples = len(y_train)

class_weights = {}

for class_id in range(NUM_CLASSES):

    weight = total_samples / (
        NUM_CLASSES * class_counts[class_id]
    )

    class_weights[class_id] = weight


print("\n====================================")
print("CLASS WEIGHTS")
print("====================================")

for class_id, name in enumerate(class_names):

    print(
        f"{name:55} "
        f"Images: {class_counts[class_id]:4d} "
        f"Weight: {class_weights[class_id]:.3f}"
    )


# ============================================================
# 8. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])


# ============================================================
# 9. BUILD CNN
# ============================================================

model = models.Sequential([

    # Input
    layers.Input(shape=(128, 128, 3)),

    # Data augmentation
    data_augmentation,

    # ----------------------------------------
    # CNN BLOCK 1
    # ----------------------------------------

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),


    # ----------------------------------------
    # CNN BLOCK 2
    # ----------------------------------------

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),


    # ----------------------------------------
    # CNN BLOCK 3
    # ----------------------------------------

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),


    # ----------------------------------------
    # CLASSIFICATION
    # ----------------------------------------

    layers.GlobalAveragePooling2D(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.5),

    # 13 classes
    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# ============================================================
# 10. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 11. SHOW MODEL
# ============================================================

print("\n====================================")
print("MODEL SUMMARY")
print("====================================")

model.summary()


# ============================================================
# 12. TRAIN CNN
# ============================================================

print("\n====================================")
print("STARTING TRAINING")
print("====================================")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights
)


# ============================================================
# 13. SAVE TRAINED MODEL
# ============================================================

model.save("plant_disease_cnn.keras")

print("\n====================================")
print("TRAINING COMPLETE")
print("====================================")

print("Model saved as:")
print("plant_disease_cnn.keras")

model.save("plant_disease_cnn.keras")