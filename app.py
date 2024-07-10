import streamlit as st
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Initialize NLTK tools
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

# Load the trained model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

processed_review = preprocess_text("Good")
review_tfidf = tfidf_vectorizer.transform([processed_review])
prediction = model.predict(review_tfidf)[0]

print(prediction)
# Streamlit UI
#st.title("IMDB Sentiment Analysis")

#review = st.text_area("Enter your review", height=200)

#if st.button("Analyze"):
#    if review:
 #       processed_review = preprocess_text(review)
#        review_tfidf = tfidf_vectorizer.transform([processed_review])
 #       prediction = model.predict(review_tfidf)[0]
 #       st.success(f"Sentiment Prediction: {'Positive' if prediction == 1 else 'Negative'}")
 #   else:
#st.warning("Please enter a review.")
