
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import pickle
from PIL import Image, ImageDraw, ImageFont
import tempfile

# 📦 Chargement du modèle et du label encoder
with open("landmarks_model_mlp.pkl", "rb") as f:
    model = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# 🔤 Fonction de prédiction
def predict_letter(image):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        flat = []
        for lm in landmarks.landmark:
            flat.extend([lm.x, lm.y, lm.z])
        X_input = np.array(flat).reshape(1, -1)
        pred = model.predict(X_input)
        letter = le.inverse_transform(pred)[0]
        return letter, landmarks
    else:
        return None, None

# 🖼️ Fonction d'annotation de l'image
def annotate_image(image, letter):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.load_default()

    text = f"Lettre : {letter}"
    text_size = draw.textbbox((0, 0), text, font=font)
    text_width = text_size[2] - text_size[0]
    text_height = text_size[3] - text_size[1]
    x = img_pil.width - text_width - 10
    y = img_pil.height - text_height - 10

    draw.rectangle([x - 5, y - 5, x + text_width + 5, y + text_height + 5], fill="white")
    draw.text((x, y), text, fill="black", font=font)
    return img_pil

# 🧠 Interface Streamlit
st.set_page_config(page_title="Reconnaissance ASL", layout="centered")
st.title("🤟 Reconnaissance de lettres en langue des signes (ASL)")

uploaded_file = st.file_uploader("📤 Charge une photo contenant une main", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Image chargée", use_column_width=True)

    with st.spinner("Analyse de la main..."):
        letter, _ = predict_letter(image)
        if letter:
            st.success(f"✅ Lettre prédite : **{letter}**")
            annotated = annotate_image(image, letter)
            st.image(annotated, caption=f"Lettre : {letter}", use_column_width=True)
        else:
            st.error("❌ Aucune main détectée. Essaie une image plus claire ou bien cadrée.")
