import streamlit as st
import joblib

from preprocess import preprocess_text

# Load model and vectorizer
# Load model and vectorizer safely
try:
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except Exception as e:
    st.error(f"Error loading model or vectorizer:\n\n{e}")
    st.stop()
# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)
st.sidebar.title("📌 About")

st.sidebar.write("""
### Dataset
IMDb 50K Movie Reviews

### Model
- Logistic Regression
- TF-IDF Vectorization"""
)

st.sidebar.info("""
This application predicts the sentiment of movie reviews using:

- TF-IDF Vectorization
- Logistic Regression
- IMDb 50K Movie Reviews Dataset
""")

# -----------------------------
# Title
# -----------------------------
st.markdown("""
<h1 style='text-align:center; color:#4CAF50;'>
🎬 Movie Review Sentiment Analysis
</h1>

<p style='text-align:center; font-size:18px;'>
Predict whether a movie review is Positive or Negative using Machine Learning.
</p>
""", unsafe_allow_html=True)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Accuracy", "89.4%")

with col2:
    st.metric("Dataset", "50K")

with col3:
    st.metric("Model", "Logistic Regression")

st.write(
    """
    Enter a movie review below and the model will predict
    whether the sentiment is **Positive** or **Negative**.
    """
)

# -----------------------------
# User Input
# -----------------------------
review = st.text_area(
    "✍️ Enter Movie Review",
    height=180,
    placeholder="Example:\nThis movie was amazing. The acting was outstanding..."
)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")

    else:

        # Preprocess review
        processed_review = preprocess_text(review)

        # TF-IDF Transformation
        review_vector = vectorizer.transform([processed_review])

        # Prediction
        with st.spinner("Analyzing review..."):

            prediction = model.predict(review_vector)[0]
            probability = model.predict_proba(review_vector)[0]

        # Probability
        probability = model.predict_proba(review_vector)[0]

        st.divider()

        st.subheader("🎯 Prediction")
        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")

        st.subheader("Confidence")

        confidence=max(probability)

        st.metric(
            "Prediction Confidence",
            f"{confidence*100:.2f}%"
        )

        st.progress(confidence)
        st.write("### Prediction Probabilities")

        st.write(f"😊 Positive: **{probability[1]*100:.2f}%**")
        st.write(f"😞 Negative: **{probability[0]*100:.2f}%**")

st.divider()


st.info(
"""
⚠ **Note**

This model uses TF-IDF and Logistic Regression.

It may not correctly understand sarcasm, irony or complex negations.
"""
)

st.caption("Developed by Divya Saini")