import streamlit as st
import pickle
from utils import clean_text

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# ─── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #0a0a0f;
        color: #e0e0f0;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 50%, #0a0a0f 100%);
    }

    .main-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99,60,255,0.15);
        border: 1px solid rgba(99,60,255,0.3);
        color: #a78bfa;
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 6px 14px;
        border-radius: 100px;
        margin-bottom: 1.2rem;
    }

    .main-title {
        font-family: 'Syne', sans-serif;
        font-size: 3.2rem;
        font-weight: 800;
        color: #f0f0ff;
        line-height: 1.05;
        letter-spacing: -0.03em;
        margin-bottom: 0.5rem;
    }

    .main-title span {
        background: linear-gradient(135deg, #a78bfa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        color: #6b6b8a;
        font-size: 0.95rem;
        font-weight: 300;
        margin-bottom: 2.5rem;
        max-width: 480px;
    }

    .section-label {
        font-size: 11px;
        font-weight: 500;
        color: #6b6b8a;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }

    .stTextArea textarea {
        background: rgba(0,0,0,0.4) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 14px !important;
        color: #e0e0f0 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 300 !important;
        line-height: 1.7 !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }

    .stTextArea textarea:focus {
        border-color: rgba(99,60,255,0.5) !important;
        box-shadow: 0 0 0 3px rgba(99,60,255,0.1) !important;
    }

    .stTextArea textarea::placeholder {
        color: #3a3a5c !important;
    }

    .stButton > button {
        width: 100%;
        padding: 1rem;
        background: linear-gradient(135deg, #6333ff, #4f46e5);
        border: none;
        border-radius: 14px;
        color: #fff;
        font-family: 'Syne', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        cursor: pointer;
        transition: transform 0.15s, box-shadow 0.15s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(99,60,255,0.4);
    }

    .result-box {
        padding: 2rem;
        border-radius: 20px;
        margin-top: 1.5rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .fake-box {
        background: rgba(220,38,38,0.08);
        border: 1px solid rgba(220,38,38,0.25);
    }

    .real-box {
        background: rgba(52,211,153,0.08);
        border: 1px solid rgba(52,211,153,0.25);
    }

    .result-label {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.3rem;
    }

    .fake-box .result-label { color: #f87171; }
    .real-box .result-label { color: #34d399; }

    .confidence-label {
        font-size: 0.85rem;
        color: #6b6b8a;
        font-weight: 300;
    }

    .confidence-label strong {
        color: #a0a0c0;
        font-weight: 500;
    }

    .tips-box {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 20px;
        padding: 1.5rem 1.8rem;
        margin-top: 2rem;
        font-size: 0.85rem;
        color: #5a5a7a;
        line-height: 1.8;
    }

    .tips-box b {
        color: #6b6b8a;
        font-weight: 500;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        font-size: 11px;
    }

    hr {
        border-color: rgba(255,255,255,0.06) !important;
        margin: 2rem 0 !important;
    }

    .stAlert {
        border-radius: 12px !important;
        border-width: 1px !important;
    }

    .stExpander {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 12px !important;
    }

    footer { display: none !important; }

    .custom-footer {
        text-align: center;
        font-size: 12px;
        color: #3a3a5c;
        margin-top: 2rem;
        padding-bottom: 2rem;
    }

    .custom-footer span { color: #6333ff; }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# ─── Header ────────────────────────────────────────────────────
st.markdown('<div class="main-badge">⚡ NLP + Machine Learning</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Fake News<br><span>Detector</span></div>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Analyze any news article instantly. Powered by advanced natural language processing and machine learning models.</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# ─── Input ─────────────────────────────────────────────────────
st.markdown('<div class="section-label">Article Input</div>', unsafe_allow_html=True)

user_input = st.text_area(
    label="",
    placeholder="Paste a news article or any text here to analyze its credibility...",
    height=200,
    label_visibility="collapsed"
)

if user_input.strip():
    word_count = len(user_input.split())
    st.caption(f"📊 Word Count: {word_count}")
    if word_count < 20:
        st.warning("⚠️ For more accurate results, please enter a longer text (20+ words recommended).")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_btn = st.button("🔍 Analyze Article")

# ─── Prediction ────────────────────────────────────────────────
if analyze_btn:
    if user_input.strip() == "":
        st.error("⚠️ Please enter some text before analyzing.")
    else:
        with st.spinner("Analyzing article, please wait..."):
            cleaned = clean_text(user_input)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            probabilities = model.predict_proba(vectorized)[0]

            fake_prob = probabilities[0] * 100
            real_prob = probabilities[1] * 100
            confidence = max(fake_prob, real_prob)

        st.markdown("---")
        st.markdown('<div class="section-label">Analysis Result</div>', unsafe_allow_html=True)

        if prediction == 0:
            st.markdown(f"""
            <div class="result-box fake-box">
                <div style="font-size:2.5rem;margin-bottom:0.8rem">❌</div>
                <div class="result-label">Fake News Detected</div>
                <div class="confidence-label">Confidence: <strong>{confidence:.1f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box real-box">
                <div style="font-size:2.5rem;margin-bottom:0.8rem">✅</div>
                <div class="result-label">Credible News</div>
                <div class="confidence-label">Confidence: <strong>{confidence:.1f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Probability Breakdown</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("❌ Fake Probability", f"{fake_prob:.1f}%")
            st.progress(fake_prob / 100)
        with col2:
            st.metric("✅ Real Probability", f"{real_prob:.1f}%")
            st.progress(real_prob / 100)

        if confidence < 65:
            st.info("🤔 The model confidence is low. We recommend verifying this article through additional trusted sources.")
        elif confidence > 90:
            st.success("💡 The model has high confidence in this prediction.")

        with st.expander("🔬 Preprocessed Text — Model Input Preview"):
            st.text(cleaned[:500] + "..." if len(cleaned) > 500 else cleaned)

# ─── Tips ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="tips-box">
    <b>Tips for Best Results</b><br><br>
    • Paste the full article body, not just the headline<br>
    • The model performs best with English-language content<br>
    • Longer texts (50+ words) yield more accurate predictions<br>
    • A confidence score above 70% indicates a reliable result
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="custom-footer">
    Built with <span>♥</span> using Streamlit & Scikit-learn
</div>
""", unsafe_allow_html=True)
