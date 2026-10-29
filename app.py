
import streamlit as st
from PIL import Image
from pathlib import Path

from railguard.predict import Predictor, ModelUnavailable
from railguard import config


# -------------------- PAGE CONFIGURATION --------------------
st.set_page_config(
    page_title="RailGuard AI",
    page_icon="🚆",
    layout="wide",
)

# -------------------- CUSTOM STYLING --------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.7rem;
        font-weight: 800;
        color: #1f4e79;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748b;
        margin-top: 0.3rem;
    }
    .result-card {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #dbe4ee;
        background-color: #f8fafc;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- MODEL LOADING --------------------
@st.cache_resource
def load_predictor():
    return Predictor(device="cpu")


# -------------------- HEADER --------------------
st.markdown(
    '<p class="main-title">🚆 RailGuard AI</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitle">AI-powered railway track defect classification</p>',
    unsafe_allow_html=True,
)

st.divider()

st.write(
    "Upload a railway track image to classify its condition using "
    "your trained deep learning models."
)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("⚙️ Prediction Settings")

    model_options = {
        "MobileNetV2": "mobilenet_v2",
        "EfficientNetB0": "efficientnet_b0",
    }

    selected_model_name = st.selectbox(
        "Choose a model",
        list(model_options.keys()),
    )

    selected_model_id = model_options[selected_model_name]

    st.info(
        "The app runs inference on the server using your trained "
        "PyTorch checkpoint."
    )

    st.caption("Supported formats: JPG, JPEG, PNG")

# -------------------- IMAGE UPLOAD --------------------
uploaded_file = st.file_uploader(
    "📤 Upload a railway track image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1, 1], gap="large")

        with left:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)
            st.caption(f"Image dimensions: {image.width} × {image.height}")

        with right:
            st.subheader("Prediction")

            if st.button("🔍 Analyze Track", type="primary", use_container_width=True):
                with st.spinner(f"Analyzing with {selected_model_name}..."):
                    try:
                        predictor = load_predictor()
                        result = predictor.predict(
                            image=image,
                            model_id=selected_model_id,
                        )

                        st.session_state["railguard_result"] = result
                        st.session_state["railguard_image_name"] = uploaded_file.name

                    except ModelUnavailable as exc:
                        st.error(f"Model unavailable: {exc}")
                    except Exception as exc:
                        st.error(
                            "Prediction failed. Check the model checkpoint "
                            "and installed dependencies."
                        )
                        st.exception(exc)

            result = st.session_state.get("railguard_result")

            if result is not None:
                if result["model_id"] != selected_model_id:
                    st.info("Select **Analyze Track** to run the selected model.")
                elif st.session_state.get("railguard_image_name") != uploaded_file.name:
                    st.info("Select **Analyze Track** to analyze this new image.")
                else:
                    st.success("Analysis completed")

                    st.markdown("### Predicted Class")
                    st.subheader(f"🚦 {result['predicted_class']}")

                    confidence = result["confidence"]
                    st.metric("Model confidence", f"{confidence:.1%}")
                    st.progress(max(0.0, min(1.0, confidence)))

                    if result["low_confidence"]:
                        st.warning(
                            result["warning"]
                            or "The prediction is uncertain. Have the track "
                            "inspected by a qualified person."
                        )
                    else:
                        st.info(
                            "This is an AI classification, not a safety "
                            "certification. Physical inspection may still be required."
                        )

                    st.caption(
                        f"Model: {result['model_name']} | "
                        f"Inference time: {result['inference_ms']:.2f} ms"
                    )

                    st.markdown("### Top 3 Predictions")
                    for item in result["top3"]:
                        st.write(
                            f"**{item['class']}** — "
                            f"{item['probability']:.1%}"
                        )
                        st.progress(
                            max(0.0, min(1.0, item["probability"]))
                        )

                    with st.expander("View all class scores"):
                        for class_name, probability in result["probabilities"].items():
                            st.write(f"{class_name}: {probability:.2%}")

    except Exception as exc:
        st.error("Unable to read this image. Please upload a valid JPG or PNG file.")
        st.caption(str(exc))

else:
    st.info("Upload an image above to get started.")

# -------------------- FOOTER --------------------
st.divider()
st.caption(
    "RailGuard AI | MobileNetV2 + EfficientNetB0 | "
    "Research prototype — not a substitute for railway safety inspection."
)
