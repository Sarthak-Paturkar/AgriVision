import os
import sys
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image


# ============================================================
# AGRIVISION - FINAL PREDICTION PIPELINE
# ============================================================
#
# Pipeline:
#
# Input Image
#      ↓
# Resize 128 × 128
#      ↓
# CNN
#      ↓
# GlobalAveragePooling2D
#      ↓
# 128-dimensional feature vector
#      ↓
# StandardScaler
#      ↓
# SVM
#      ↓
# Final Disease Class
#
# IMPORTANT:
# The SVM is the FINAL classifier.
# CNN softmax predictions are shown only as a diagnostic.
#
# ============================================================


# ============================================================
# 1. SETTINGS
# ============================================================

IMG_SIZE = (128, 128)

CNN_MODEL_PATH = "plant_disease_cnn.keras"

ML_MODEL_PATH = "data/ml_models/best_model.joblib"

CLASS_NAMES_PATH = "data/features/class_names.npy"

# Default image if no image path is supplied
DEFAULT_IMAGE_PATH = "test_leaf.png"

EXPECTED_FEATURES = 128

NUM_CLASSES = 13


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def print_separator(title):
    print("\n" + "=" * 44)
    print(title)
    print("=" * 44)


def check_file(path, description):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )

    print(f"OK  {description}: {path}")


def clean_class_name(name):
    """
    Makes dataset class names easier to read.
    """
    return str(name).replace("___", " -> ")


# ============================================================
# 3. DETERMINE INPUT IMAGE
# ============================================================

if len(sys.argv) > 1:
    IMAGE_PATH = sys.argv[1]
else:
    IMAGE_PATH = DEFAULT_IMAGE_PATH


# ============================================================
# 4. CHECK REQUIRED FILES
# ============================================================

print_separator("CHECKING REQUIRED FILES")

check_file(CNN_MODEL_PATH, "CNN model")
check_file(ML_MODEL_PATH, "ML model")
check_file(CLASS_NAMES_PATH, "Class names")
check_file(IMAGE_PATH, "Input image")


# ============================================================
# 5. LOAD CLASS NAMES
# ============================================================

print_separator("LOADING CLASS NAMES")

class_names = np.load(
    CLASS_NAMES_PATH,
    allow_pickle=True
).tolist()

class_names = [str(name) for name in class_names]

print("Number of classes:", len(class_names))

if len(class_names) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes, "
        f"but found {len(class_names)}."
    )

for i, name in enumerate(class_names):
    print(f"{i}: {name}")


# ============================================================
# 6. LOAD CNN
# ============================================================

print_separator("LOADING CNN")

cnn = tf.keras.models.load_model(
    CNN_MODEL_PATH
)

print("CNN loaded successfully.")


# ============================================================
# 7. VERIFY CNN INPUT SHAPE
# ============================================================

print_separator("VERIFYING CNN INPUT")

try:
    cnn_input_shape = tuple(cnn.input_shape)
except Exception:
    cnn_input_shape = None

print("CNN input shape:", cnn_input_shape)

expected_input_shape = (
    None,
    IMG_SIZE[0],
    IMG_SIZE[1],
    3
)

if cnn_input_shape is not None:
    if cnn_input_shape != expected_input_shape:
        raise ValueError(
            "\nCNN input shape mismatch!\n"
            f"Expected : {expected_input_shape}\n"
            f"Received : {cnn_input_shape}"
        )

print("CNN input verification: PASSED")


# ============================================================
# 8. INITIALIZE CNN SAFELY
# ============================================================

print_separator("INITIALIZING CNN")

dummy_input = tf.zeros(
    (1, IMG_SIZE[0], IMG_SIZE[1], 3),
    dtype=tf.float32
)

# Explicit forward pass.
# This guarantees the Sequential model has been built.
_ = cnn(dummy_input, training=False)

print("CNN initialized successfully.")


# ============================================================
# 9. LOCATE GLOBAL AVERAGE POOLING LAYER
# ============================================================

print_separator("LOCATING FEATURE LAYER")

FEATURE_LAYER_NAME = "global_average_pooling2d"

try:
    feature_layer = cnn.get_layer(
        FEATURE_LAYER_NAME
    )

except ValueError:

    print(
        f"ERROR: Feature layer "
        f"'{FEATURE_LAYER_NAME}' was not found."
    )

    print("\nAvailable CNN layers:")

    for i, layer in enumerate(cnn.layers):
        print(
            f"{i}: {layer.name} "
            f"({layer.__class__.__name__})"
        )

    raise


print("Feature layer found successfully.")
print("Layer name :", feature_layer.name)
print("Layer type :", feature_layer.__class__.__name__)


# ============================================================
# 10. CREATE FEATURE EXTRACTOR
# ============================================================

print_separator("CREATING FEATURE EXTRACTOR")

# Use the actual tensor created by the model.
#
# cnn.input can fail with certain Sequential/Keras
# configurations if the model has not been properly built.
#
# The dummy forward pass above ensures that cnn.inputs
# is available.

if not hasattr(cnn, "inputs") or len(cnn.inputs) == 0:
    raise RuntimeError(
        "CNN input tensor is unavailable after initialization."
    )

feature_extractor = tf.keras.Model(
    inputs=cnn.inputs[0],
    outputs=feature_layer.output
)

print("Feature extractor created successfully.")

print(
    "Feature output shape:",
    feature_extractor.output_shape
)


# ============================================================
# 11. VERIFY FEATURE DIMENSION
# ============================================================

feature_dimension = feature_extractor.output_shape[-1]

print("\nFeature dimension:", feature_dimension)

if feature_dimension != EXPECTED_FEATURES:
    raise ValueError(
        "\nFeature dimension mismatch!\n"
        f"Expected : {EXPECTED_FEATURES}\n"
        f"Received : {feature_dimension}"
    )

print("Confirmed: 128 features per image.")


# ============================================================
# 12. LOAD FINAL SVM MODEL
# ============================================================

print_separator("LOADING FINAL ML MODEL")

ml_model = joblib.load(
    ML_MODEL_PATH
)

print("ML model loaded successfully.")

print(
    "Model type :",
    type(ml_model).__name__
)


# ============================================================
# 13. VERIFY SVM CLASS ORDER
# ============================================================

if not hasattr(ml_model, "classes_"):

    raise RuntimeError(
        "The saved ML model does not expose "
        "classes_. Cannot safely verify class ordering."
    )

ml_classes = np.asarray(
    ml_model.classes_
).astype(int)

print("\nML model classes:")
print(ml_classes)

expected_classes = np.arange(
    len(class_names)
)

if not np.array_equal(
    ml_classes,
    expected_classes
):

    raise ValueError(
        "\nCLASS ORDER MISMATCH!\n"
        f"ML model classes : {ml_classes}\n"
        f"Expected classes : {expected_classes}"
    )

print("ML class ordering verified successfully.")


# ============================================================
# 14. LOAD IMAGE
# ============================================================

print_separator("LOADING IMAGE")

print("Image path:", IMAGE_PATH)

try:

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

except Exception as e:

    raise RuntimeError(
        f"Could not open image:\n{IMAGE_PATH}\n\n"
        f"Original error: {e}"
    )


print(
    "Original image size:",
    image.size
)


# ============================================================
# 15. RESIZE IMAGE
# ============================================================

image = image.resize(
    IMG_SIZE,
    Image.Resampling.BILINEAR
)

print(
    "Resized image size:",
    image.size
)


# ============================================================
# 16. CONVERT IMAGE TO NUMPY
# ============================================================

image_array = np.asarray(
    image,
    dtype=np.float32
)

print(
    "Image array shape:",
    image_array.shape
)

print("\nPixel range before CNN:")
print(
    "Minimum :",
    f"{image_array.min():.2f}"
)

print(
    "Maximum :",
    f"{image_array.max():.2f}"
)


# ============================================================
# 17. ADD BATCH DIMENSION
# ============================================================

image_batch = np.expand_dims(
    image_array,
    axis=0
)

print(
    "\nCNN input shape:",
    image_batch.shape
)


# ============================================================
# 18. VERIFY INPUT
# ============================================================

expected_shape = (
    1,
    IMG_SIZE[0],
    IMG_SIZE[1],
    3
)

if image_batch.shape != expected_shape:

    raise ValueError(
        "\nInvalid CNN input shape!\n"
        f"Expected : {expected_shape}\n"
        f"Received : {image_batch.shape}"
    )


# ============================================================
# 19. CNN DIRECT PREDICTION
# ============================================================
#
# This is NOT the final classifier.
#
# It is shown only because it helps us inspect what the CNN
# itself thinks before the 128-D features are passed into SVM.
#
# The previous ~43% result came from this stage.
# ============================================================

print_separator("CNN DIRECT PREDICTION")

cnn_output = cnn(
    image_batch,
    training=False
).numpy()

cnn_output = np.asarray(
    cnn_output
).reshape(-1)

if len(cnn_output) != NUM_CLASSES:

    print(
        "CNN output is not a 13-class probability vector."
    )

    cnn_probabilities = None

else:

    # If the model's final layer already outputs probabilities,
    # these values should sum approximately to 1.
    cnn_sum = float(
        np.sum(cnn_output)
    )

    print(
        "CNN output values:",
        len(cnn_output)
    )

    print(
        "CNN output sum:",
        f"{cnn_sum:.6f}"
    )

    # Determine whether output already looks like probabilities.
    if (
        np.all(cnn_output >= 0.0)
        and
        np.all(cnn_output <= 1.0)
        and
        np.isclose(cnn_sum, 1.0, atol=1e-4)
    ):

        cnn_probabilities = cnn_output

    else:

        # Treat output as logits and convert to probabilities.
        cnn_probabilities = tf.nn.softmax(
            cnn_output
        ).numpy()


    cnn_top_indices = np.argsort(
        cnn_probabilities
    )[-3:][::-1]

    print("\nCNN TOP 3:")

    for rank, index in enumerate(
        cnn_top_indices,
        start=1
    ):

        index = int(index)

        print(
            f"{rank}. "
            f"{class_names[index]} : "
            f"{cnn_probabilities[index] * 100:.2f}%"
        )


# ============================================================
# 20. CNN FEATURE EXTRACTION
# ============================================================

print_separator("EXTRACTING CNN FEATURES")

features = feature_extractor(
    image_batch,
    training=False
).numpy()

print(
    "Raw feature shape:",
    features.shape
)


# ============================================================
# 21. FLATTEN FEATURE VECTOR
# ============================================================

features = features.reshape(
    features.shape[0],
    -1
)

print(
    "Final feature shape:",
    features.shape
)


# ============================================================
# 22. VERIFY FEATURE VECTOR
# ============================================================

if features.shape != (
    1,
    EXPECTED_FEATURES
):

    raise ValueError(
        "\nUnexpected feature shape!\n"
        f"Expected : (1, {EXPECTED_FEATURES})\n"
        f"Received : {features.shape}"
    )

print(
    "Feature verification: PASSED"
)


# ============================================================
# 23. GENERATE FINAL SVM PREDICTION
# ============================================================

print_separator(
    "GENERATING FINAL SVM PREDICTION"
)

predicted_index = int(
    ml_model.predict(
        features
    )[0]
)

print(
    "Predicted class index:",
    predicted_index
)


# ============================================================
# 24. VERIFY PREDICTED CLASS INDEX
# ============================================================

if not (
    0 <= predicted_index < len(class_names)
):

    raise ValueError(
        f"Invalid predicted class index: "
        f"{predicted_index}"
    )


predicted_class = class_names[
    predicted_index
]


# ============================================================
# 25. GET SVM PROBABILITIES
# ============================================================

svm_probabilities = None

if hasattr(
    ml_model,
    "predict_proba"
):

    try:

        svm_probabilities = np.asarray(
            ml_model.predict_proba(
                features
            )[0],
            dtype=np.float64
        )

    except Exception as e:

        print(
            "\nWARNING:"
            "\nSVM probability calculation failed."
            f"\nReason: {e}"
        )

        svm_probabilities = None


# ============================================================
# 26. VERIFY SVM PROBABILITIES
# ============================================================

if svm_probabilities is not None:

    if len(svm_probabilities) != len(
        class_names
    ):

        raise ValueError(
            "\nSVM probability vector size "
            "does not match number of classes.\n"
            f"Probabilities : "
            f"{len(svm_probabilities)}\n"
            f"Classes       : "
            f"{len(class_names)}"
        )

    probability_sum = float(
        np.sum(svm_probabilities)
    )

    print(
        "\nProbability sanity check:"
    )

    print(
        "Number of probabilities :",
        len(svm_probabilities)
    )

    print(
        "Probability sum         :",
        f"{probability_sum:.6f}"
    )

    if not np.isclose(
        probability_sum,
        1.0,
        atol=1e-5
    ):

        raise ValueError(
            "SVM probabilities do not sum to 1."
        )


# ============================================================
# 27. TOP 3 SVM PREDICTIONS
# ============================================================

print_separator(
    "FINAL SVM TOP 3 PREDICTIONS"
)

if svm_probabilities is not None:

    top_k = min(
        3,
        len(svm_probabilities)
    )

    top_indices = np.argsort(
        svm_probabilities
    )[-top_k:][::-1]

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        index = int(index)

        print(
            f"{rank}. "
            f"{class_names[index]} : "
            f"{svm_probabilities[index] * 100:.2f}%"
        )


else:

    print(
        "SVM probabilities are unavailable."
    )


# ============================================================
# 28. FINAL SVM CONFIDENCE
# ============================================================
#
# IMPORTANT:
#
# The FINAL predicted class comes from:
#
#     ml_model.predict()
#
# Therefore the displayed SVM probability must correspond
# to THAT predicted class.
#
# We do NOT simply take the largest probability and call it
# the predicted class.
# ============================================================

if svm_probabilities is not None:

    confidence = float(
        svm_probabilities[
            predicted_index
        ]
    ) * 100.0

else:

    confidence = None


# ============================================================
# 29. FINAL RESULT
# ============================================================

print_separator(
    "FINAL PREDICTION RESULT"
)

print(
    "Predicted class :",
    predicted_class
)

if confidence is not None:

    print(
        f"Confidence      : "
        f"{confidence:.2f}%"
    )

else:

    print(
        "Confidence      : "
        "Not available"
    )


# ============================================================
# 30. CNN VS SVM COMPARISON
# ============================================================

print_separator(
    "CNN VS FINAL SVM"
)

if cnn_probabilities is not None:

    cnn_index = int(
        np.argmax(
            cnn_probabilities
        )
    )

    cnn_confidence = (
        float(
            cnn_probabilities[
                cnn_index
            ]
        ) * 100.0
    )

    print(
        "CNN direct prediction :",
        class_names[cnn_index]
    )

    print(
        f"CNN probability       : "
        f"{cnn_confidence:.2f}%"
    )

else:

    print(
        "CNN direct prediction : unavailable"
    )


print(
    "Final SVM prediction  :",
    predicted_class
)

if confidence is not None:

    print(
        f"SVM probability       : "
        f"{confidence:.2f}%"
    )


# ============================================================
# 31. FINAL SANITY CHECK
# ============================================================

print_separator(
    "FINAL SANITY CHECK"
)

print(
    "Number of classes     :",
    len(class_names)
)

print(
    "Feature dimension     :",
    features.shape[1]
)

print(
    "Predicted class index :",
    predicted_index
)

print(
    "CNN feature extractor : OK"
)

print(
    "Feature dimension     : OK"
)

print(
    "ML class ordering     : OK"
)

if svm_probabilities is not None:
    print(
        "Probability vector    : OK"
)

print(
    "Prediction pipeline   : OK"
)


# ============================================================
# 32. PIPELINE SUMMARY
# ============================================================

print_separator(
    "AGRIVISION PREDICTION PIPELINE"
)

print(
    "Input Image"
    " -> 128x128 RGB"
    " -> CNN"
    " -> 128-D Features"
    " -> StandardScaler"
    " -> SVM"
    " -> Disease Class"
)

print(
    "\nFINAL CLASSIFIER: CNN + SVM"
)

print(
    "\nCNN is used as the feature extractor."
)

print(
    "SVM is used as the final classifier."
)


# ============================================================
# 33. COMPLETE
# ============================================================

print_separator(
    "PREDICTION COMPLETE"
)