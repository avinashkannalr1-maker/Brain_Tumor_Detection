import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🧠 AI-Based Brain Tumor Detection")
st.write(
    "Upload a brain MRI image to classify it using our "
    "VGG16-based deep learning model."
)

st.info(
    "The system classifies MRI images into four categories: "
    "Glioma, Meningioma, No Tumor, and Pituitary Tumor."
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

MODEL_PATH = "best_vgg16_brain_tumor.keras"

@st.cache_resource
def load_brain_tumor_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_brain_tumor_model()
    st.success("✅ VGG16 model loaded successfully.")
except Exception as e:
    st.error("❌ Could not load the trained model.")
    st.write(e)
    st.stop()

# --------------------------------------------------
# CLASS NAMES
# --------------------------------------------------

class_names = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

def preprocess_image(image):

    # Convert PIL image to OpenCV format
    image = np.array(image)

    # Convert RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Convert back to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to 224 x 224
    image = cv2.resize(image, (224, 224))

    # Gaussian noise reduction
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # CLAHE contrast enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    enhanced_lab = cv2.merge((l, a, b))

    image = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2RGB
    )

    # Normalization
    image = image.astype(np.float32) / 255.0

    return image

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

st.subheader("📤 Upload MRI Image")

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("🖼️ Uploaded MRI Image")

    st.image(
        image,
        caption="Uploaded MRI Image",
        use_container_width=True
    )

    # Prediction button
    if st.button("🔍 Predict Tumor Type"):

        with st.spinner("Analyzing MRI image..."):

            # Preprocess
            processed_image = preprocess_image(image)

            # Add batch dimension
            input_image = np.expand_dims(
                processed_image,
                axis=0
            )

            # Prediction
            prediction = model.predict(
                input_image,
                verbose=0
            )

            # Find predicted class
            predicted_class = np.argmax(
                prediction[0]
            )

            confidence = (
                prediction[0][predicted_class] * 100
            )

            predicted_label = class_names[
                predicted_class
            ]

        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        st.subheader("📋 Prediction Result")

        st.success(
            f"Predicted Class: **{predicted_label}**"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        # --------------------------------------------------
        # PROBABILITY DISTRIBUTION
        # --------------------------------------------------

        st.subheader("📊 Class Probability")

        probabilities = prediction[0] * 100

        probability_data = {
            class_names[i]: float(probabilities[i])
            for i in range(len(class_names))
        }

        st.bar_chart(probability_data)

        # --------------------------------------------------
        # VOICE ASSIST
        # --------------------------------------------------

        st.subheader("🔊 Voice Assist")

        voice_text = (
            f"The predicted class is {predicted_label}, "
            f"with a confidence of {confidence:.2f} percent. "
            "This result is intended for research purposes "
            "and should not replace professional medical diagnosis."
        )

        # Escape text safely for JavaScript
        import json

        safe_text = json.dumps(voice_text)

        voice_button = f"""
        <script>
        function speakResult() {{
            const text = {safe_text};
            const speech = new SpeechSynthesisUtterance(text);

            speech.lang = "en-US";
            speech.rate = 0.9;
            speech.pitch = 1.0;

            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(speech);
        }}
        </script>

        <button
            onclick="speakResult()"
            style="
                background-color:#ff4b4b;
                color:white;
                border:none;
                padding:12px 20px;
                border-radius:8px;
                font-size:16px;
                cursor:pointer;
            "
        >
            🔊 Speak Result
        </button>
        """

        st.components.v1.html(
            voice_button,
            height=60
        )

        # --------------------------------------------------
        # DISCLAIMER
        # --------------------------------------------------

        st.warning(
            "⚠️ Research Prototype: This AI system is intended "
            "for academic and research purposes only. It should "
            "not be used as a substitute for professional "
            "medical diagnosis."
        )