import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Loading the dataset from a csv file
def loading_dataset(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocess the comments for machine learning
def preprocess_data(data):
    stop_words = stopwords.words('english')
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(data['comment'])
    y = data['Sentiment']
    return X, y, vectorizer

# Model training
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    #Model evaluation
    y_pred = model.predict(X_test)
    st.write("Model Performance:")
    st.text(classification_report(y_test, y_pred))
    
    return model

# Predict sentiment for given comments
def predict_sentiment(model, vectorizer, comments):
    comments_transformed = vectorizer.transform(comments)
    predictions = model.predict(comments_transformed)
    return predictions

# Visualize the sentiment distribution
def sentiment_distribution_plot(data):
    sentiment_count = data['Sentiment'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(sentiment_count, labels=sentiment_count.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
    ax.axis('equal')  
    st.pyplot(fig)