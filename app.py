import streamlit as st
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Initialize NLTK tools
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
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

# Streamlit UI
st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬", layout="wide")

# Sidebar
st.sidebar.header("About the App")
st.sidebar.write("""
This app uses a machine learning model to analyze the sentiment of your movie review. 
Enter a review in the text box below and click "Analyze" to see the sentiment analysis.
""")

st.sidebar.header("How it Works")
st.sidebar.write("""
1. The review is preprocessed to remove HTML tags, punctuation, and stopwords, and then stemmed.
2. The preprocessed review is transformed into a TF-IDF vector.
3. The model predicts the sentiment of the review.
""")


st.sidebar.header("About the Model")
st.sidebar.write("""
The model is trained on 50,000 IMDB movie reviews. I am yet to optimize is further for accuracy. So, this might show some inaccuracies.
""")

# Add links to the sidebar
st.sidebar.header("Useful Links")
st.sidebar.markdown("[Link to the Notebook](https://github.com/Ashfinn/IMDB-Sentiment-Analysis/blob/main/notebook.ipynb)")
st.sidebar.markdown("[GitHub Repository](https://github.com/Ashfinn/IMDB-Sentiment-Analysis)")
st.sidebar.markdown("[Portfolio](https://ashfinn.github.io/)")

# Main content
st.title("IMDB Sentiment Analysis 🎬")
st.write("Enter a review below to analyze its sentiment (positive or negative).")

review = st.text_area("Enter your review", height=200)

if st.button("Analyze"):
    if review:
        with st.spinner('Analyzing the sentiment...'):
            processed_review = preprocess_text(review)
            review_tfidf = tfidf_vectorizer.transform([processed_review])
            sentiment = model.predict(review_tfidf)[0]
            st.success(f"Sentiment Prediction: **{sentiment}**")
    else:
        st.warning("Please enter a review.")

# Custom CSS to style the app
st.markdown("""
    <style>
    .css-1d391kg {
        max-width: 800px;
        margin: auto;
    }
    .css-1aehpvz {
        margin: auto;
    }
    .css-1d391kg h1 {
        text-align: center;
    }
    .css-1d391kg textarea {
        font-size: 16px;
    }
    .css-1d391kg button {
        font-size: 18px;
        padding: 10px 20px;
        background-color: #4CAF50;
        color: white;
    }
    .css-1d391kg .stAlert {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
