import streamlit as st
import joblib

model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

st.title("📰 Fake News Detector")

news = st.text_area("Enter News Article")

if st.button("Check"):
    transformed = vectorizer.transform([news])
    prediction = model.predict(transformed)

    if prediction[0] == 1:
        st.success("Real News")
    else:
        st.error("Fake News")
