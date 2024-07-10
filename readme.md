# IMDB Sentiment Analysis 🎬

This Streamlit app uses a machine learning model to analyze the sentiment. This app is designed to predict the sentiment of your texts as either positive or negative. It uses a pre-trained machine learning model and a TF-IDF vectorizer trained on 50,000 IMDB reviews.

## Features

- Preprocessing of text data
- Sentiment analysis using a trained machine learning model
- Web interface using Streamlit
- Includes a Jupyter Notebook for model training and evaluation

## How It Works
1. **Preprocessing**: The input review is preprocessed to remove HTML tags, punctuation, and stopwords, and then stemmed.
2. **TF-IDF Transformation**: The preprocessed review is transformed into a TF-IDF vector.
3. **Prediction**: The model predicts the sentiment of the text based on the TF-IDF vector.

## Jupyter Notebook

The [Notebook](notebook.ipynb) file contains the code for training and evaluating the sentiment analysis model. You can open and run this notebook to see how the model was developed.

## Project Structure
- **app.py:** Main file for running the Streamlit web app.
- **IMDB Dataset.csv:** Dataset used for training the model.
- **model.pkl:** Trained sentiment analysis model.
- **notebook.ipynb:** Jupyter Notebook with model training and evaluation code.
- **requirements.txt:** List of required Python packages.
- **tfidf_vectorizer.pkl:** TF-IDF vectorizer used for text preprocessing.

## Contributing
Contributions are welcome! Please fork the repository and use a feature branch. Pull requests are warmly welcome.

## Acknowledgments
- NLTK
- scikit-learn
- Streamlit