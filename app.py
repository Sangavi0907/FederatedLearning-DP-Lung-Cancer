#  Streamlit and Core Utilities
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import pickle

# TensorFlow / Keras (for building and loading models)
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.models import load_model

# Optional: For advanced metrics
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


st.set_page_config(layout="wide")
st.title("🏥 Federated Lung Cancer Prediction - Dashboard ")

# Load saved synthetic images if available
SYN_IMG_PATH = "synthetic_images.npy"
if os.path.exists(SYN_IMG_PATH):
    synthetic_images = np.load(SYN_IMG_PATH)
else:
    st.markdown("Images not available!!")

# Sidebar
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", [
    "Overview",
    "Visualize GAN",
    "Metrics Comparison",
    "Federated vs Centralized", 
    "Prediction Demo"
])


# ======================== 🔍 SECTION 1: OVERVIEW ========================
if section == "Overview":
    st.header("📌 Project Summary")
    st.markdown("""
    **Objective:**  
    Detect lung cancer using both CT scan images and patient metadata through a secure and privacy-preserving federated learning approach.

    **Key Features:**

    - Federated Learning using fog nodes for decentralized training.
    - Multimodal model combining Convolutional Neural Networks (CNN) for image data and Feedforward Neural Networks (FFNN) for structured metadata.
    - Enhanced privacy using Generalization and Threshold Shuffler algorithms.
    - Adversarial training integrated to improve robustness and confidentiality.
    - Synthetic data augmentation via GANs (Generative Adversarial Networks).
    - Secure model weight transfer and aggregation using AWS S3.

    This architecture ensures patient privacy while maintaining model performance across distributed nodes.
    """)


# ======================== 🎨 SECTION 2: GAN IMAGES ========================
elif section == "Visualize GAN":
    st.header("📄 GAN-Generated Lung CT Images")
    st.caption("These are synthetic samples generated to augment data.")

    num = st.slider("Pick Image Index", 0, len(synthetic_images)-1, 0)
    img = synthetic_images[num].squeeze()

    #Convert [0,1] → [0,255] and uint8
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    st.image(img, caption=f"Synthetic Image #{num}", width=256)


    st.success("GAN helps protect privacy + improve accuracy by augmenting data.")

# ======================== 📊 SECTION 3: METRIC COMPARISON ========================

elif section == "Metrics Comparison":
    st.header("📊 Performance with vs without Adversarial Training")

    import json

    try:
        with open("metrics.json", "r") as f:
            m = json.load(f)

        labels = ['Accuracy', 'Loss', 'AUC']
        clean = [m["accuracy_clean"], m["loss_clean"], m["auc_clean"]]
        adv   = [m["accuracy_adv"], m["loss_adv"], m["auc_adv"]]

        st.success("✅ Metrics loaded from model evaluation.")

    except Exception as e:
        st.warning("⚠️ Could not load `metrics.json`, using default demo values.")
        labels = ['Accuracy', 'Loss', 'AUC', 'NMI']
        clean = [0.78, 0.46, 0.84, 0.10]
        adv   = [0.82, 0.42, 0.87, 0.09]

    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    ax.bar(x - 0.2, clean, width=0.2, label="Without Adv Training", color='skyblue')
    ax.bar(x + 0.2, adv, width=0.2, label="With Adv Training", color='lightgreen')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Scores")
    ax.set_ylim(0, 1)
    ax.set_title("Metric Comparison")
    ax.legend()

    for i in range(len(labels)):
        ax.text(i - 0.2, clean[i] + 0.02, f"{clean[i]:.2f}", ha='center', fontsize=9)
        ax.text(i + 0.2, adv[i] + 0.02, f"{adv[i]:.2f}", ha='center', fontsize=9)

    st.pyplot(fig)

    if 'clean' in locals():
        delta_acc = adv[0] - clean[0]
        st.success(f"🎯 Accuracy ↑ by {delta_acc:.2f} with adversarial training.")


# ======================== 📈 SECTION 4: Federated vs Centralized ========================
elif section == "Federated vs Centralized":
    st.header("📈 Federated Learning vs Centralized Learning")
    st.caption("Comparison of key metrics for both training approaches")

    try:
        metric_data = np.load("comparison_metrics.npy", allow_pickle=True).item()
        central = metric_data['centralized']
        federated = metric_data['federated']

        labels = ['Accuracy', 'Loss', 'AUC']
        x = np.arange(len(labels))

        fig, ax = plt.subplots()
        ax.bar(x - 0.2, central, width=0.4, label="Centralized", color='orange')
        ax.bar(x + 0.2, federated, width=0.4, label="Federated", color='royalblue')

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.1)
        ax.set_title("Training Strategy Comparison")
        ax.legend()
        st.pyplot(fig)

        st.success("✅ Centralized models may perform slightly better, but federated learning provides **privacy** and **distributed training**.")

    except Exception as e:
        st.error(f"❌ Failed to load comparison data: {e}")



# ======================== 🔍 SECTION 5: PREDICTION DEMO ========================
elif section == "Prediction Demo":
    st.header("🔍 Try a Lung Cancer Prediction Sample")

    uploaded_img = st.file_uploader("📤 Upload a Lung CT Scan (Grayscale)", type=["jpg", "png"])
    
    st.markdown("✅ Fill in your health metadata below:")

    # Full metadata input: 15 features
    meta_values = []
    meta_features = [
        "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", "CHRONIC DISEASE",
        "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL CONSUMING", "COUGHING", "SHORTNESS OF BREATH",
        "SWALLOWING DIFFICULTY", "CHEST PAIN"
    ]

    cols = st.columns(3)
    for i, feature in enumerate(meta_features):
        with cols[i % 3]:
            val = st.selectbox(f"{feature}", [1, 0], key=feature) if feature != "AGE" else st.slider("AGE", 0, 100, 50)
            meta_values.append(val)

    meta_input = np.array([meta_values])

    if uploaded_img:
        img_array = cv2.imdecode(np.frombuffer(uploaded_img.read(), np.uint8), 0)
        img_array = cv2.resize(img_array, (128, 128)).reshape(1, 128, 128, 1) / 255.0
        st.image(img_array.squeeze(), caption="📷 Uploaded CT Image", width=256)

        st.warning("🚀 Loading model and making prediction...")
        try:
            with open("final_global_model_5.pkl", "rb") as f:
                weights = pickle.load(f)

            def build_cnn_ffnn_binary(image_shape, metadata_shape):
                img_input = keras.Input(shape=image_shape)
                x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
                x = keras.layers.MaxPooling2D((2, 2))(x)
                x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
                x = keras.layers.MaxPooling2D((2, 2))(x)
                x = keras.layers.Flatten()(x)
                x = keras.layers.Dense(128, activation='relu')(x)
                meta_input = keras.Input(shape=(15,))
                m = keras.layers.Dense(64, activation='relu')(meta_input)
                m = keras.layers.Dense(32, activation='relu')(m)
                combined = keras.layers.concatenate([x, m])
                z = keras.layers.Dense(64, activation='relu')(combined)
                output = keras.layers.Dense(1, activation='sigmoid')(z)
                model = keras.Model(inputs=[img_input, meta_input], outputs=output)
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                return model

            model = build_cnn_ffnn_binary((128, 128, 1), 15)
            model.set_weights(weights)
            pred = model.predict([img_array, meta_input])[0][0]
            pred_class = "Lung Cancer Detected" if pred > 0.5 else "Normal"
            st.success(f"✅ Prediction: **{pred_class}** (Confidence: {pred:.2f})")
        except Exception as e:
            st.error(f"❌ Failed to load model or predict: {e}")


