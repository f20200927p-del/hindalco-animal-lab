"""Beginner-friendly Streamlit client for the deployed animal model."""

import base64
import io
import os

import pandas as pd
import requests
import streamlit as st
from PIL import Image


def jpeg_base64(image):
    """Shrink the uploaded image and return its JPEG bytes as base64 text."""
    image = image.convert("RGB")
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=85, optimize=True)
    return base64.b64encode(output.getvalue()).decode("ascii")


st.set_page_config(page_title="Animal Predictor")
st.title("Animal Predictor")

endpoint_url = os.environ.get("ENDPOINT_URL")
endpoint_key = os.environ.get("ENDPOINT_KEY")
uploaded = st.file_uploader("Upload an animal photo", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded photo", use_container_width=True)

    if not endpoint_url or not endpoint_key:
        st.error("Set ENDPOINT_URL and ENDPOINT_KEY before making a prediction.")
    elif st.button("Predict"):
        body = {"image": jpeg_base64(image)}
        try:
            response = requests.post(
                endpoint_url,
                json=body,
                headers={"Authorization": f"Bearer {endpoint_key}"},
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            if "error" in result:
                st.error(result["error"])
            else:
                confidence = float(result["confidence"])
                st.subheader(f"Animal: {result['animal']}")
                st.write(f"Confidence: {confidence:.1%}")
                scores = pd.DataFrame(
                    {"animal": list(result["all_scores"]), "score": list(result["all_scores"].values())}
                ).set_index("animal")
                st.bar_chart(scores)
                if confidence < 0.60:
                    st.warning("The model is not very confident in this prediction.")
        except requests.RequestException as error:
            st.error(f"Request failed: {error}")