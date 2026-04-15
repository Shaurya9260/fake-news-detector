import streamlit as st
import pickle
from utils import clean_text

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detector")

user_input = st.text_area("Enter News Text:")

if st.button("Analyze"):
    if user_input.strip() != "":
        
        # 🔥 SAME CLEANING AS TRAINING
        cleaned_text = clean_text(user_input)
        
        # Transform
        vector_input = vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(vector_input)[0]
        
        if prediction == 0:
            st.error("❌ Fake News")
        else:
            st.success("✅ Real News")
    else:
        st.warning("Please enter some text")