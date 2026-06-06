# main.py
from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
import google.generativeai as genai
from PIL import Image

# --- Configure API key ---
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError(
        "Missing API key. Create a .env file with GOOGLE_API_KEY=your_key"
    )
genai.configure(api_key=API_KEY)

# --- Auto-detect best available model ---
PREFERRED_MODELS = [
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-lite",
    "models/gemini-1.5-pro",
    "models/gemini-1.5-flash",
    "models/gemini-1.0-pro",
]

def get_best_model() -> str:
    try:
        available = [
            m.name for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]
        for preferred in PREFERRED_MODELS:
            if preferred in available:
                return preferred
        # fallback: pick first available multimodal model
        if available:
            return available[0]
    except Exception as e:
        st.warning(f"Could not auto-detect model: {e}. Falling back to gemini-2.0-flash.")
    return "models/gemini-2.0-flash"

MODEL_NAME = get_best_model()
model = genai.GenerativeModel(MODEL_NAME)


def get_gemini_response(prompt: str, image_part: dict | None = None) -> str:
    try:
        parts = [prompt]
        if image_part:
            if not isinstance(image_part, dict) or "data" not in image_part:
                raise ValueError("image_part must be a dict with keys 'mime_type' and 'data'")
            parts.append(image_part)

        resp = model.generate_content(parts)
        text = getattr(resp, "text", None)
        return (text or "").strip() or "No textual response returned by the model."

    except Exception as e:
        msg = str(e)
        if "API key not valid" in msg or "API_KEY_INVALID" in msg:
            raise RuntimeError(
                "API key invalid. Re-check the key in your .env or generate a new one at https://aistudio.google.com/apikey"
            ) from e
        if "not found" in msg and "model" in msg.lower():
            raise RuntimeError(
                f"Model '{MODEL_NAME}' is not available for your key.\n"
                f"Run genai.list_models() to see what's available."
            ) from e
        raise


def prepare_image_part(uploaded_file) -> dict | None:
    if uploaded_file is None:
        return None
    data = uploaded_file.getvalue()
    if not data:
        raise FileNotFoundError("Uploaded file is empty or unreadable.")
    mime = uploaded_file.type or (
        "image/png" if uploaded_file.name.lower().endswith(".png") else "image/jpeg"
    )
    return {"mime_type": mime, "data": data}


# --- Streamlit UI ---
st.set_page_config(page_title="Nutrition App", page_icon="🥗")
st.header("Nutrition App 👨‍⚕️")
st.caption(f"Using model: `{MODEL_NAME}`")  # helpful for debugging

uploaded_file = st.file_uploader("Choose an image of your meal", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image.", use_column_width=True)

submit = st.button("Tell me about my meal")

input_prompt = """
You are an expert nutritionist. Inspect the food items from the image and calculate the total calories.
Provide details of every food item with calories in this format:

1. Item 1 - no of calories
2. Item 2 - no of calories
----

After that mention whether the meal is healthy or not and mention the percentage split of:
- Carbohydrates
- Proteins
- Fats
- Sugar
- Total Calories

Finally give suggestions:
- Which items should be removed to make the meal healthier
- Which items should be added to improve nutrition
"""

if submit:
    if uploaded_file is None:
        st.error("Please upload an image before clicking the button.")
    else:
        with st.spinner("Analyzing your meal..."):
            try:
                image_part = prepare_image_part(uploaded_file)
                response = get_gemini_response(input_prompt, image_part)
                st.subheader("Nutritional Analysis")
                st.write(response)
            except Exception as e:
                st.error(f"Error: {e}")