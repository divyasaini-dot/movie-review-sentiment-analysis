import streamlit as st
import joblib

from preprocess import preprocess_text

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

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
st.title("🎬 Movie Review Sentiment Analysis")

st.markdown("""
Analyze the sentiment of a movie review using a **Machine Learning model**
trained on the **IMDb 50K Movie Reviews Dataset**.

**Model Used:** Logistic Regression + TF-IDF
""")

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
    "Enter Movie Review",
    height=200,
    placeholder="Example: This movie was absolutely fantastic!"
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
        prediction = model.predict(review_vector)[0]

        # Probability
        probability = model.predict_proba(review_vector)[0]

        st.divider()

        st.subheader("Prediction")

        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")

        st.subheader("Confidence")

        confidence = max(probability) * 100

        st.metric("Confidence", f"{confidence:.2f}%")

        st.progress(float(max(probability)))

        st.write("### Prediction Probabilities")

        st.write(f"😊 Positive: **{probability[1]*100:.2f}%**")
        st.write(f"😞 Negative: **{probability[0]*100:.2f}%**")

st.divider()

st.caption("Developed by Divya Saini")