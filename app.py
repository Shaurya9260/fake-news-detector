import streamlit as st
import pickle
from utils import clean_text

# Load
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="AI Fake News Detector", page_icon="🧠")

st.title("🧠 Fake News Detection System")
st.write("Check whether news is Fake or Real")

news_input = st.text_area("Enter News", height=200)

if st.button("Analyze"):
    if news_input.strip() == "":
        st.warning("Please enter text")
    else:
        with st.spinner("Analyzing..."):
            cleaned = clean_text(news_input)
            vectorized = vectorizer.transform([cleaned])

            prediction = model.predict(vectorized)
            prob = model.predict_proba(vectorized)

            confidence = max(prob[0])

            if prediction[0] == 1:
                st.success(f"✅ REAL (Confidence: {confidence:.2f})")
            else:
                st.error(f"❌ FAKE (Confidence: {confidence:.2f})")