# ============================================================
# AGRIVISION - PLANT DISEASE DETECTION
# CNN FEATURE EXTRACTOR + SVM CLASSIFIER
# ============================================================

import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AgriVision",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CNN_PATH = os.path.join(
    BASE_DIR,
    "plant_disease_cnn.keras"
)

ML_MODEL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "ml_models",
    "best_model.joblib"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "class_names.npy"
)

IMAGE_SIZE = (128, 128)
FEATURE_DIMENSION = 128


# ============================================================
# CLASS NAMES
# ============================================================

EXPECTED_CLASSES = [
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clean_class_name(class_name):
    """
    Convert dataset class names into human-readable labels.

    Example:
        Tomato___Early_blight
        ->
        Tomato - Early Blight
    """

    name = str(class_name)

    parts = name.split("___", 1)

    if len(parts) == 2:
        plant = parts[0]
        disease = parts[1]
    else:
        plant = name
        disease = ""

    disease = disease.replace("_", " ")
    disease = disease.replace("(", " (")
    disease = " ".join(disease.split())

    return plant, disease


def is_healthy_class(class_name):
    """
    Determine whether the predicted class represents
    a healthy leaf.
    """

    return str(class_name).lower().endswith("___healthy")


def format_confidence(value):
    return f"{float(value) * 100:.2f}%"


# ============================================================
# LOAD CLASS NAMES
# ============================================================

@st.cache_resource
def load_class_names():

    if not os.path.exists(CLASS_NAMES_PATH):
        raise FileNotFoundError(
            f"Class names file not found:\n{CLASS_NAMES_PATH}"
        )

    loaded = np.load(CLASS_NAMES_PATH, allow_pickle=True)

    classes = [str(x) for x in loaded.tolist()]

    if len(classes) != len(EXPECTED_CLASSES):
        raise ValueError(
            f"Expected {len(EXPECTED_CLASSES)} classes, "
            f"but found {len(classes)}."
        )

    # Verify ordering because prediction indices depend on it.
    if classes != EXPECTED_CLASSES:
        raise ValueError(
            "Class ordering mismatch detected.\n\n"
            "The class_names.npy file does not match the "
            "class ordering used by the trained model."
        )

    return classes


# ============================================================
# LOAD CNN
# ============================================================

@st.cache_resource
def load_cnn():

    if not os.path.exists(CNN_PATH):
        raise FileNotFoundError(
            f"CNN model not found:\n{CNN_PATH}"
        )

    model = keras.models.load_model(
        CNN_PATH,
        compile=False
    )

    return model


# ============================================================
# INITIALIZE CNN
# ============================================================

@st.cache_resource
def initialize_cnn():

    cnn = load_cnn()

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    # A loaded Sequential model can exist without having been
    # called in the current process. Calling it once explicitly
    # prevents:
    #
    # "The layer sequential_1 has never been called..."
    #
    # from appearing when creating the feature extractor.
    # --------------------------------------------------------

    dummy_input = tf.zeros(
        (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        dtype=tf.float32
    )

    _ = cnn(dummy_input, training=False)

    return cnn


# ============================================================
# CREATE FEATURE EXTRACTOR
# ============================================================

@st.cache_resource
def create_feature_extractor():

    cnn = initialize_cnn()

    feature_layer = None

    # --------------------------------------------------------
    # First preference:
    # GlobalAveragePooling2D
    # --------------------------------------------------------

    for layer in cnn.layers:

        if isinstance(
            layer,
            keras.layers.GlobalAveragePooling2D
        ):
            feature_layer = layer
            break

    # --------------------------------------------------------
    # Fallback:
    # Find the layer whose output has 128 features.
    # --------------------------------------------------------

    if feature_layer is None:

        for layer in reversed(cnn.layers):

            try:
                output_shape = layer.output.shape

                if (
                    len(output_shape) == 2
                    and output_shape[-1] == FEATURE_DIMENSION
                ):
                    feature_layer = layer
                    break

            except Exception:
                continue

    if feature_layer is None:
        raise RuntimeError(
            "Could not locate the 128-dimensional feature layer "
            "inside the CNN."
        )

    # --------------------------------------------------------
    # The CNN has already been called above, so its input is
    # now defined and can safely be used here.
    # --------------------------------------------------------

    feature_extractor = keras.Model(
        inputs=cnn.inputs,
        outputs=feature_layer.output,
        name="agrivision_feature_extractor"
    )

    # Verify feature dimension.
    test_features = feature_extractor.predict(
        np.zeros(
            (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
            dtype=np.float32
        ),
        verbose=0
    )

    if test_features.shape[-1] != FEATURE_DIMENSION:
        raise RuntimeError(
            f"Unexpected feature dimension: "
            f"{test_features.shape[-1]}. "
            f"Expected {FEATURE_DIMENSION}."
        )

    return feature_extractor


# ============================================================
# LOAD SVM
# ============================================================

@st.cache_resource
def load_ml_model():

    if not os.path.exists(ML_MODEL_PATH):
        raise FileNotFoundError(
            f"ML model not found:\n{ML_MODEL_PATH}"
        )

    model = joblib.load(ML_MODEL_PATH)

    return model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(uploaded_image):

    # --------------------------------------------------------
    # Convert explicitly to RGB.
    #
    # This prevents RGBA / grayscale / palette images from
    # producing the wrong channel structure.
    # --------------------------------------------------------

    image = Image.open(uploaded_image).convert("RGB")

    original_image = image.copy()

    # --------------------------------------------------------
    # Resize exactly to the CNN input size.
    # --------------------------------------------------------

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT divide by 255 here unless the trained CNN itself
    # was trained with external /255 preprocessing.
    #
    # Your previous successful pipeline passed pixel values
    # in the 0-255 range to the CNN, and the trained model
    # handles its own expected preprocessing.
    # --------------------------------------------------------

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Add batch dimension.
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return original_image, image_array


# ============================================================
# PREDICTION
# ============================================================

def predict_leaf(image_array):

    feature_extractor = create_feature_extractor()
    ml_model = load_ml_model()
    class_names = load_class_names()

    # --------------------------------------------------------
    # CNN -> 128-D FEATURES
    # --------------------------------------------------------

    features = feature_extractor.predict(
        image_array,
        verbose=0
    )

    features = np.asarray(
        features,
        dtype=np.float32
    )

    # Ensure shape is exactly (1, 128).
    if features.ndim != 2:
        features = features.reshape(
            features.shape[0],
            -1
        )

    if features.shape != (1, FEATURE_DIMENSION):
        raise RuntimeError(
            f"Unexpected CNN feature shape: "
            f"{features.shape}. "
            f"Expected (1, {FEATURE_DIMENSION})."
        )

    # --------------------------------------------------------
    # 128-D FEATURES -> SVM
    # --------------------------------------------------------

    predicted_indices = ml_model.predict(features)

    predicted_index = int(
        predicted_indices[0]
    )

    # --------------------------------------------------------
    # Validate prediction index.
    # --------------------------------------------------------

    if predicted_index < 0 or predicted_index >= len(class_names):
        raise RuntimeError(
            f"Invalid predicted class index: "
            f"{predicted_index}"
        )

    predicted_class = class_names[predicted_index]

    # --------------------------------------------------------
    # SVM PROBABILITY
    # --------------------------------------------------------

    if not hasattr(ml_model, "predict_proba"):
        raise RuntimeError(
            "The saved SVM model does not provide "
            "predict_proba()."
        )

    probabilities = ml_model.predict_proba(features)[0]

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Verify probability vector.
    # --------------------------------------------------------

    if len(probabilities) != len(class_names):
        raise RuntimeError(
            "Probability vector length does not match "
            "the number of classes."
        )

    probability_sum = probabilities.sum()

    if not np.isfinite(probability_sum):
        raise RuntimeError(
            "Invalid probability values returned by SVM."
        )

    # Normalize only for numerical precision.
    if not np.isclose(
        probability_sum,
        1.0,
        atol=1e-5
    ):
        probabilities = (
            probabilities / probability_sum
        )

    confidence = float(
        probabilities[predicted_index]
    )

    # --------------------------------------------------------
    # Sort probabilities for optional top predictions.
    #
    # These are NOT displayed as separate CNN predictions.
    # They are all probabilities from the SAME final SVM.
    # --------------------------------------------------------

    ranked_indices = np.argsort(
        probabilities
    )[::-1]

    top_predictions = []

    for index in ranked_indices[:3]:

        top_predictions.append({
            "index": int(index),
            "class_name": class_names[index],
            "probability": float(
                probabilities[index]
            )
        })

    return {
        "class_index": predicted_index,
        "class_name": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "top_predictions": top_predictions,
    }


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🌿 AgriVision")

st.subheader(
    "AI-Powered Plant Disease Detection"
)

st.write(
    "Upload a leaf image and analyze it to identify "
    "the plant condition."
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ],
    help="Upload a clear image of a plant leaf."
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # Read image for display.
    # --------------------------------------------------------

    try:

        preview_image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception:

        st.error(
            "The uploaded file could not be read as an image."
        )

        st.stop()

    # --------------------------------------------------------
    # Display uploaded image.
    # --------------------------------------------------------

    st.markdown("### Uploaded Leaf")

    st.image(
        preview_image,
        caption="Uploaded leaf image",
        width=500
    )

    st.write("")

    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    analyze = st.button(
        "🔍 Analyze Leaf",
        type="primary",
        use_container_width=True
    )

    if analyze:

        with st.spinner(
            "Analyzing the leaf..."
        ):

            try:

                # Reset file pointer.
                uploaded_file.seek(0)

                # --------------------------------------------
                # PREPROCESS
                # --------------------------------------------

                original_image, image_array = (
                    preprocess_image(
                        uploaded_file
                    )
                )

                # --------------------------------------------
                # FINAL PREDICTION
                # --------------------------------------------

                result = predict_leaf(
                    image_array
                )

            except Exception as error:

                st.error(
                    "Unable to analyze this image."
                )

                with st.expander(
                    "Technical error details"
                ):
                    st.exception(error)

                st.stop()

        # ====================================================
        # FINAL RESULT
        # ====================================================

        class_name = result["class_name"]
        confidence = result["confidence"]

        plant_name, disease_name = clean_class_name(
            class_name
        )

        healthy = is_healthy_class(
            class_name
        )

        st.divider()

        st.markdown(
            "## Analysis Result"
        )

        # ----------------------------------------------------
        # HEALTHY RESULT
        # ----------------------------------------------------

        if healthy:

            st.success(
                "🌱 Healthy Leaf"
            )

            st.markdown(
                f"### {plant_name} leaf"
            )

            st.write(
                f"Confidence: **{format_confidence(confidence)}**"
            )

            st.info(
                f"The model identified this as a healthy "
                f"{plant_name.lower()} leaf."
            )

        # ----------------------------------------------------
        # DISEASE RESULT
        # ----------------------------------------------------

        else:

            st.error(
                "🦠 Disease Detected"
            )

            st.markdown(
                f"### {plant_name} — {disease_name}"
            )

            st.write(
                f"Confidence: **{format_confidence(confidence)}**"
            )

            st.warning(
                f"The model identified signs consistent with "
                f"**{disease_name}** on a **{plant_name}** leaf."
            )

        # ====================================================
        # TOP PREDICTIONS
        # ====================================================
        #
        # These are ONLY the final SVM's probabilities.
        # We do not display CNN predictions anywhere.
        # ====================================================

        with st.expander(
            "View alternative predictions"
        ):

            for rank, prediction in enumerate(
                result["top_predictions"],
                start=1
            ):

                p_plant, p_disease = clean_class_name(
                    prediction["class_name"]
                )

                if is_healthy_class(
                    prediction["class_name"]
                ):
                    label = f"{p_plant} — Healthy"
                else:
                    label = f"{p_plant} — {p_disease}"

                st.write(
                    f"**{rank}. {label}** — "
                    f"{format_confidence(prediction['probability'])}"
                )

                st.progress(
                    min(
                        max(
                            float(
                                prediction["probability"]
                            ),
                            0.0
                        ),
                        1.0
                    )
                )

        # ====================================================
        # SIMPLE DISCLAIMER
        # ====================================================

        st.caption(
            "AgriVision provides an AI-based image classification "
            "result. For important agricultural decisions, "
            "consider confirmation from a qualified agricultural "
            "professional."
        )


# ============================================================
# NO IMAGE STATE
# ============================================================

else:

    st.info(
        "👆 Upload a leaf image above to begin analysis."
    )