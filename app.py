import streamlit as st
import joblib
import re
import numpy as np
from scipy.sparse import hstack

# -----------------------------
# Load saved models & objects
# -----------------------------
BASE_PATH = r"C:\Users\sahit\OneDrive\Desktop\ACM Project"

tfidf = joblib.load(f"{BASE_PATH}\\tfidf.pkl")
scaler = joblib.load(f"{BASE_PATH}\\scaler.pkl")
clf = joblib.load(f"{BASE_PATH}\\classification_model.pkl")
reg = joblib.load(f"{BASE_PATH}\\regression_model.pkl")

# -----------------------------
# Text cleaning (same as training)
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s+\-*/=<>]", "", text)
    return text.strip()

# -----------------------------
# STEP 2: Feature Extraction
# -----------------------------
def extract_features(text):
    cleaned = clean_text(text)

    # TF-IDF features
    tfidf_features = tfidf.transform([cleaned])

    # Custom features
    text_length = len(cleaned.split())
    math_symbol_count = len(re.findall(r"[+\-*/=<>]", cleaned))

    custom_features = np.array([[text_length, math_symbol_count]])
    custom_features = scaler.transform(custom_features)

    return hstack([tfidf_features, custom_features])

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AutoJudge", layout="centered")

st.title("🧠 AutoJudge: Difficulty Predictor")
st.write("Paste the programming problem details below:")

problem_desc = st.text_area("Problem Description")
input_desc = st.text_area("Input Description")
output_desc = st.text_area("Output Description")

if st.button("Predict"):
    combined_text = problem_desc + " " + input_desc + " " + output_desc

    features = extract_features(combined_text)

    class_pred = clf.predict(features)[0]
    score_pred = reg.predict(features)[0]

    class_map = {0: "Easy", 1: "Medium", 2: "Hard"}

    st.subheader("🔍 Prediction Results")
    st.write(f"**Predicted Difficulty Class:** {class_map[class_pred]}")
    st.write(f"**Predicted Difficulty Score:** {score_pred:.2f}")
