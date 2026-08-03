# 🎬 Movie Review Sentiment Analysis

A Machine Learning web application that predicts whether a movie review is **Positive** or **Negative** using **Natural Language Processing (NLP)** techniques.

The project uses **TF-IDF Vectorization** for feature extraction and **Logistic Regression** for sentiment classification. A user-friendly web interface is built using **Streamlit**.

---

## 🚀 Live Demo

> *(Add your Streamlit deployment link here after deployment.)*

Example:

https://your-app-name.streamlit.app

---

## 📌 Features

- Predicts sentiment of movie reviews
- Text preprocessing using NLTK
- TF-IDF feature extraction
- Logistic Regression classifier
- Interactive Streamlit web application
- Displays prediction confidence
- Easy-to-use interface

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Joblib
- Streamlit

---

## 📂 Dataset

**IMDb 50K Movie Reviews Dataset**

- 50,000 movie reviews
- Balanced dataset
- Positive and Negative sentiment labels

Source:
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

---

## 📊 Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. TF-IDF Feature Extraction
4. Train/Test Split
5. Logistic Regression Model Training
6. Model Evaluation
7. Hyperparameter Tuning
8. Streamlit Deployment

---

## ⚙️ Model Performance

| Metric | Score |
|---------|--------|
| Accuracy | **88.75%** |
| Precision | **88.76%** |
| Recall | **88.75%** |
| F1-Score | **88.75%** |

---

## 📸 Application Preview

*(Add screenshots here after deployment.)*

Example:

### Home Page

![Home](screenshots/home.png)

### Prediction Result

![Prediction](screenshots/result.png)

---

## 📁 Project Structure

```text
Sentiment-Analysis/
│
├── app.py
├── preprocess.py
├── sentiment_analysis.ipynb
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
├── README.md
└── screenshots/
```

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Sentiment-Analysis.git
```

Move into the project folder

```bash
cd Sentiment-Analysis
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 💡 Example Reviews

### Positive

> This movie was absolutely fantastic. I loved every scene.

Prediction

✅ Positive

---

### Negative

> This movie was boring and a complete waste of time.

Prediction

❌ Negative

---

## 📈 Future Improvements

- Fine-tune a BERT model for higher accuracy
- Add Neutral sentiment classification
- Improve preprocessing with lemmatization
- Support multilingual sentiment analysis
- Deploy using Docker and cloud platforms

---

## 📚 Learning Outcomes

This project demonstrates:

- Natural Language Processing (NLP)
- Text preprocessing
- TF-IDF Vectorization
- Logistic Regression
- Hyperparameter tuning using GridSearchCV
- Model serialization with Joblib
- Web application deployment using Streamlit

---

## 👨‍💻 Author

**Divya Saini**

GitHub: https://github.com/divyasaini

LinkedIn: https://www.linkedin.com/in/divya-saini-76a198383/

---

## ⭐ If you like this project

Please consider giving it a ⭐ on GitHub!