import streamlit as st
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- Main background ---------- */

    .stApp {
        background-color: #0e1117;
    }

    /* ---------- Sidebar ---------- */

    [data-testid="stSidebar"] {
        background-color: #151922;
        border-right: 1px solid #2a2f3a;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff;
    }

    /* ---------- Main title ---------- */

    .main-title {
        font-size: 48px;
        font-weight: 800;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 5px;
        color: #ffffff;
        letter-spacing: -1px;
    }

    .main-title span {
        color: #49d17d;
    }

    .subtitle {
        text-align: center;
        color: #aab2c0;
        font-size: 18px;
        margin-bottom: 35px;
    }

    /* ---------- Cards ---------- */

    .metric-card {
        background: linear-gradient(
            145deg,
            #171b24,
            #11141b
        );

        border: 1px solid #2b313d;
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        min-height: 130px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.18);
    }

    .metric-title {
        color: #9da6b5;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 30px;
        font-weight: 700;
    }

    .metric-description {
        color: #6f7888;
        font-size: 12px;
        margin-top: 5px;
    }

    /* ---------- Section headers ---------- */

    .section-title {
        color: #ffffff;
        font-size: 24px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    /* ---------- Prediction cards ---------- */

    .positive-card {
        background: linear-gradient(
            135deg,
            rgba(55, 190, 108, 0.16),
            rgba(25, 80, 50, 0.12)
        );

        border: 1px solid rgba(73, 209, 125, 0.45);
        border-radius: 18px;
        padding: 30px;
        text-align: center;
    }

    .negative-card {
        background: linear-gradient(
            135deg,
            rgba(230, 70, 70, 0.16),
            rgba(90, 30, 30, 0.12)
        );

        border: 1px solid rgba(255, 90, 90, 0.45);
        border-radius: 18px;
        padding: 30px;
        text-align: center;
    }

    .prediction-label {
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .prediction-confidence {
        color: #aab2c0;
        font-size: 15px;
    }

    /* ---------- Probability boxes ---------- */

    .probability-box {
        background-color: #171b24;
        border: 1px solid #2b313d;
        border-radius: 12px;
        padding: 15px 20px;
        margin-bottom: 10px;
    }

    .probability-label {
        color: #dce1e8;
        font-size: 15px;
        margin-bottom: 5px;
    }

    .probability-value {
        color: #ffffff;
        font-size: 21px;
        font-weight: 700;
    }

    /* ---------- Info card ---------- */

    .info-card {
        background-color: #151922;
        border: 1px solid #2b313d;
        border-radius: 16px;
        padding: 22px;
        margin-top: 15px;
    }

    .info-card h3 {
        color: #ffffff;
        margin-top: 0;
    }

    .info-card p {
        color: #aab2c0;
        line-height: 1.6;
    }

    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #697384;
        font-size: 13px;
        margin-top: 45px;
        padding-top: 20px;
        border-top: 1px solid #292e38;
    }

    /* ---------- Buttons ---------- */

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 45px;
    }

    /* ---------- Text area ---------- */

    textarea {
        border-radius: 12px !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL AND VECTORIZER
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    return model, vectorizer


try:

    model, vectorizer = load_model()

except Exception as e:

    st.error(
        f"""
        Unable to load the sentiment model or TF-IDF vectorizer.

        Error:
        {e}
        """
    )

    st.stop()


# Import preprocessing function
from preprocess import preprocess_text


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🎬 About Project")

    st.markdown("---")

    st.markdown("### 📚 Dataset")

    st.write("IMDb 50K Movie Reviews")

    st.markdown("### 🔤 Feature Extraction")

    st.write("TF-IDF Vectorization")

    st.markdown("### 🤖 Machine Learning Model")

    st.write("Logistic Regression")

    st.markdown("### ⚙️ Optimization")

    st.write("GridSearchCV")

    st.markdown("### 🎯 Best Parameters")

    st.code(
        "C = 10\nsolver = liblinear",
        language="text"
    )

    st.markdown("---")

    st.markdown("### 📊 Test Performance")

    st.write("Accuracy: **91.26%**")

    st.write("F1-score: **≈ 0.91**")

    st.markdown("---")

    st.markdown("### ⚠️ Limitations")

    st.caption(
        """
        The model may have difficulty with sarcasm,
        irony, ambiguous language, and very complex
        expressions.
        """
    )

    st.markdown("---")

    st.caption("Developed with ❤️ by Divya Saini")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        🎬 Movie Review <span>Sentiment Analysis</span>
    </div>

    <div class="subtitle">
        Analyze movie reviews using Natural Language Processing
        and Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL PERFORMANCE CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">MODEL ACCURACY</div>
            <div class="metric-value">91.26%</div>
            <div class="metric-description">Test dataset</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">DATASET</div>
            <div class="metric-value">50K</div>
            <div class="metric-description">IMDb movie reviews</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">FEATURES</div>
            <div class="metric-value">TF-IDF</div>
            <div class="metric-description">1–3 gram features</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:

    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">CLASSIFIER</div>
            <div class="metric-value">Logistic</div>
            <div class="metric-description">Logistic Regression</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# REVIEW INPUT
# ============================================================

st.markdown(
    '<div class="section-title">✍️ Analyze a Movie Review</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter a movie review below and the trained model will "
    "predict whether the sentiment is **Positive** or **Negative**."
)


review = st.text_area(
    "Movie Review",
    height=180,
    placeholder=(
        "Example:\n"
        "This movie was absolutely fantastic. "
        "The acting was amazing and I loved every scene."
    ),
    label_visibility="collapsed"
)


# ============================================================
# EXAMPLE REVIEWS
# ============================================================

st.caption("💡 Try an example:")

example_col1, example_col2, example_col3 = st.columns(3)

with example_col1:

    if st.button(
        "😊 Positive Example",
        use_container_width=True
    ):

        st.session_state.review_text = (
            "This movie was absolutely fantastic. "
            "I loved every scene."
        )

with example_col2:

    if st.button(
        "😞 Negative Example",
        use_container_width=True
    ):

        st.session_state.review_text = (
            "This movie was boring and terrible. "
            "I hated it."
        )

with example_col3:

    if st.button(
        "🔄 Negation Example",
        use_container_width=True
    ):

        st.session_state.review_text = (
            "I don't like this movie."
        )


# Use example text if selected
if "review_text" in st.session_state:

    review = st.session_state.review_text

    st.text_area(
        "Selected Example",
        value=review,
        height=120,
        disabled=True
    )


# ============================================================
# BUTTONS
# ============================================================

button_col1, button_col2, button_col3 = st.columns(
    [1, 1, 3]
)

with button_col1:

    analyze = st.button(
        "🔍 Analyze Sentiment",
        type="primary",
        use_container_width=True
    )

with button_col2:

    clear = st.button(
        "🗑️ Clear",
        use_container_width=True
    )

if clear:

    st.session_state.pop("review_text", None)

    st.rerun()


# ============================================================
# PREDICTION
# ============================================================

if analyze:

    if not review or not review.strip():

        st.warning(
            "⚠️ Please enter a movie review before analyzing."
        )

    else:

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        processed_review = preprocess_text(review)

        # ----------------------------------------------------
        # TF-IDF transformation
        # ----------------------------------------------------

        review_vector = vectorizer.transform(
            [processed_review]
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with st.spinner("Analyzing your review..."):

            prediction = model.predict(
                review_vector
            )[0]

            probability = model.predict_proba(
                review_vector
            )[0]

        confidence = max(probability)

        positive_probability = probability[1] * 100
        negative_probability = probability[0] * 100


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown("---")

        st.markdown(
            '<div class="section-title">🎯 Sentiment Result</div>',
            unsafe_allow_html=True
        )


        result_col1, result_col2 = st.columns(
            [1.3, 1]
        )


        # ----------------------------------------------------
        # Prediction card
        # ----------------------------------------------------

        with result_col1:

            if prediction == 1:

                st.markdown(
                    f"""
                    <div class="positive-card">

                        <div class="prediction-label">
                            😊 POSITIVE REVIEW
                        </div>

                        <div class="prediction-confidence">
                            The model predicts a positive sentiment
                        </div>

                        <br>

                        <div style="
                            font-size:42px;
                            font-weight:800;
                            color:#49d17d;
                        ">
                            {confidence * 100:.2f}%
                        </div>

                        <div class="prediction-confidence">
                            Prediction Confidence
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="negative-card">

                        <div class="prediction-label">
                            😞 NEGATIVE REVIEW
                        </div>

                        <div class="prediction-confidence">
                            The model predicts a negative sentiment
                        </div>

                        <br>

                        <div style="
                            font-size:42px;
                            font-weight:800;
                            color:#ff7070;
                        ">
                            {confidence * 100:.2f}%
                        </div>

                        <div class="prediction-confidence">
                            Prediction Confidence
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # ----------------------------------------------------
        # Probabilities
        # ----------------------------------------------------

        with result_col2:

            st.markdown("### 📊 Prediction Probabilities")

            st.markdown(
                f"""
                <div class="probability-box">

                    <div class="probability-label">
                        😊 Positive
                    </div>

                    <div class="probability-value">
                        {positive_probability:.2f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                min(
                    positive_probability / 100,
                    1.0
                )
            )


            st.markdown(
                f"""
                <div class="probability-box">

                    <div class="probability-label">
                        😞 Negative
                    </div>

                    <div class="probability-value">
                        {negative_probability:.2f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                min(
                    negative_probability / 100,
                    1.0
                )
            )


        # ====================================================
        # PROCESSED REVIEW
        # ====================================================

        st.markdown("---")

        st.markdown(
            '<div class="section-title">🔍 NLP Preprocessing</div>',
            unsafe_allow_html=True
        )

        st.caption(
            "This is the text after applying the same preprocessing "
            "pipeline used during model training."
        )

        st.code(
            processed_review if processed_review else
            "[No meaningful tokens remaining]",
            language="text"
        )


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        with st.expander("⚙️ View Model Details"):

            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:

                st.markdown("#### 🔤 Text Processing")

                st.write("• Lowercase conversion")

                st.write("• HTML removal")

                st.write("• Contraction handling")

                st.write("• Stop-word removal")

                st.write("• Negation handling")


            with detail_col2:

                st.markdown("#### 🤖 Machine Learning")

                st.write("• TF-IDF Vectorization")

                st.write("• N-grams: 1–3")

                st.write("• Logistic Regression")

                st.write("• GridSearchCV")

                st.write("• Best C: 10")


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🔬 How the System Works</div>',
    unsafe_allow_html=True
)

step1, step2, step3, step4 = st.columns(4)

with step1:

    st.markdown(
        """
        ### 01 📚
        **Input**

        A movie review is entered by the user.
        """
    )

with step2:

    st.markdown(
        """
        ### 02 🧹
        **Preprocessing**

        The text is cleaned and prepared
        using the NLP preprocessing pipeline.
        """
    )

with step3:

    st.markdown(
        """
        ### 03 🔤
        **TF-IDF**

        The processed text is converted
        into numerical features.
        """
    )

with step4:

    st.markdown(
        """
        ### 04 🤖
        **Prediction**

        Logistic Regression classifies
        the review as Positive or Negative.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        🎬 Movie Review Sentiment Analysis
        <br>
        Built with Python • NLTK • Scikit-learn • Streamlit
        <br><br>
        Developed with ❤️ by Divya Saini

    </div>
    """,
    unsafe_allow_html=True
)