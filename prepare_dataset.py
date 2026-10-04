from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split


# ============================================================
# 1. ORIGINAL DATASET
# ============================================================

DATASET_DIR = Path(
    r"C:\Users\sarth\Downloads\Plant_leaf_diseases_dataset_without_augmentation\Plant_leave_diseases_dataset_without_augmentation"
)


# ============================================================
# 2. WHERE WE WANT THE NEW DATASET
# ============================================================

OUTPUT_DIR = Path("data/plant_disease")


# ============================================================
# 3. OUR 13 CLASSES
# ============================================================

classes = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",

    "Potato___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",

    "Corn___healthy",
    "Corn___Common_rust",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",

    "Grape___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"
]


# ============================================================
# 4. TRAIN / VALIDATION / TEST FOLDERS
# ============================================================

for split in ["train", "validation", "test"]:
    for class_name in classes:
        (OUTPUT_DIR / split / class_name).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# 5. PROCESS EACH CLASS
# ============================================================

for class_name in classes:

    print(f"\nProcessing: {class_name}")

    class_folder = DATASET_DIR / class_name

    # Get image files
    images = [
        file for file in class_folder.iterdir()
        if file.is_file()
    ]

    # --------------------------------------------------------
    # First split:
    # 70% TRAIN
    # 30% TEMPORARY
    # --------------------------------------------------------

    train_images, temp_images = train_test_split(
        images,
        test_size=0.30,
        random_state=42
    )

    # --------------------------------------------------------
    # Second split:
    # TEMPORARY → 15% VALIDATION + 15% TEST
    # --------------------------------------------------------

    validation_images, test_images = train_test_split(
        temp_images,
        test_size=0.50,
        random_state=42
    )

    print(f"Train      : {len(train_images)}")
    print(f"Validation : {len(validation_images)}")
    print(f"Test       : {len(test_images)}")

    # --------------------------------------------------------
    # Copy images
    # --------------------------------------------------------

    for image in train_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "train" / class_name / image.name
        )

    for image in validation_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "validation" / class_name / image.name
        )

    for image in test_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "test" / class_name / image.name
        )


print("\n====================================")
print("DATASET PREPARATION COMPLETE")
print("====================================")