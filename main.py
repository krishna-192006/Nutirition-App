# main.py
from dotenv import load_dotenv
load_dotenv()  # reads .env in project root

import os
import streamlit as st
import google.generativeai as genai
from PIL import Image

# --- Configure API key (support both names) ---
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError(
        "Missing API key. Create a .env file with GOOGLE_API_KEY=your_key (or API_KEY=your_key)."
    )
genai.configure(api_key=API_KEY)

# --- Create model once and reuse ---
MODEL_NAME = "gemini-1.5-flash"  # or "gemini-1.5-pro"
model = genai.GenerativeModel(MODEL_NAME)


def get_gemini_response(prompt: str, image_part: dict | None = None) -> str:
    """
    prompt: text prompt
    image_part: dict like {"mime_type": "image/png", "data": b'...'} or None
    """
    try:
        parts = [prompt]
        if image_part:
            # ensure structure is correct
            if not isinstance(image_part, dict) or "data" not in image_part:
                raise ValueError("image_part must be a dict with keys 'mime_type' and 'data'")
            parts.append(image_part)

        resp = model.generate_content(parts)  # SDK returns an object with .text
        text = getattr(resp, "text", None)
        return (text or "").strip() or "No textual response returned by the model."
    except Exception as e:
        # Provide clearer messages for common problems
        msg = str(e)
        if "API key not valid" in msg or "API_KEY_INVALID" in msg:
            raise RuntimeError(
                "API key invalid. Re-check the key in your .env or generate a new one in Google AI Studio."
            ) from e
        if "not found" in msg and "model" in msg.lower():
            raise RuntimeError(
                f"Model {MODEL_NAME} may not be available to your key. Try running genai.list_models() to see available models."
            ) from e
        raise


def prepare_image_part(uploaded_file) -> dict:
    """
    Convert a Streamlit UploadedFile to the image part structure the SDK accepts.
    """
    if uploaded_file is None:
        return None

    data = uploaded_file.getvalue()  # bytes
    if not data:
        raise FileNotFoundError("Uploaded file is empty or unreadable.")

    # Use mime type from uploaded_file or infer from filename
    mime = uploaded_file.type or (
        "image/png" if uploaded_file.name.lower().endswith(".png") else "image/jpeg"
    )

    return {"mime_type": mime, "data": data}


# --- Streamlit UI ---
st.set_page_config(page_title="Nutrion App")
st.header("Nutrion App 👨‍⚕️")

uploaded_file = st.file_uploader("Choose an image..", type=["jpg", "jpeg", "png"])
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
After that mention whether the meal is healthy or not and mention the percentage split of carbohydrates, proteins, fats, sugar, and total calories in the meal.
Finally give suggestions which items should be removed and which should be added to make the meal healthy if it's unhealthy.
"""

if submit:
    try:
        if uploaded_file is None:
            st.error("Please upload an image before clicking Generate.")
        else:
            image_part = prepare_image_part(uploaded_file)
            response = get_gemini_response(input_prompt, image_part)
            st.subheader("The Response is")
            st.write(response)
    except Exception as e:
        st.error(f"Error: {e}")
