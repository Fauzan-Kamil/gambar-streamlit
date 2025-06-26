import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import test_openai
import os


# Config
st.set_page_config(page_title="Deteksi Penyakit Daun Jagung", layout="centered")

# API KEY OpenAI (gunakan .env di produksi)
test_openai.api_key = "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"  # Ganti dengan API key Anda

# Load model deteksi
model = tf.keras.models.load_model("cnn_jgg_500.h5")

# Label kelas
class_names = ['Bercak Daun', 'Busuk Daun', 'Karat Daun', 'Sehat']

# Tampilan
st.title("🌽 Deteksi Penyakit Daun Jagung Otomatis")
st.write("📷 Upload gambar daun jagung, dan sistem akan memprediksi jenis penyakit serta memberikan saran pengobatan.")

# Upload gambar
uploaded_file = st.file_uploader("Upload gambar daun jagung (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

# Function
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    st.image(image, caption="🖼️ Gambar yang Anda upload", use_column_width=True)

    # Preprocessing
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediksi dengan model
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Hasil prediksi
    st.markdown(f"### 🧪 Hasil Deteksi:")
    st.success(f"📌 Penyakit Terdeteksi: **{predicted_class}**")
    st.info(f"🎯 Tingkat Akurasi Model: **{confidence:.2f}%**")

# Rekomendasi Obat

    if predicted_class != "Sehat":
        st.markdown("### 💊 Rekomendasi Pengobatan:")
        with st.spinner("Mencari rekomendasi pengobatan dari AI..."):
            try:
                prompt = f"Berikan rekomendasi pengobatan atau obat yang sesuai untuk tanaman jagung yang terkena penyakit '{predicted_class}'. Singkat dan jelas, maksimal 3 kalimat."

                response = test_openai.ChatCompletion.create(
                    model="gpt-3",
                    messages=[
                        {"role": "system", "content": "Kamu adalah ahli pertanian dan penyakit tanaman."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,
                    temperature=0.7
                )

                rekomendasi_ai = response['choices'][0]['message']['content']
                st.success(rekomendasi_ai)

            except Exception:
                st.warning("⚠️ Gagal mengambil data")

    else:
        st.success("🌿 Tanaman Anda sehat. Tidak perlu pengobatan.")
