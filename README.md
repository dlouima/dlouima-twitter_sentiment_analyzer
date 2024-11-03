## PlayMood AI Sentiment Analysis App getting started guide

1. Prerequisites
* Python 3.7+ installed on your system.
* Git installed (for cloning the repository from GitHub).
* pip (Python package manager) to install required packages.
  
If you don’t have these installed, you can download them from the following links:
* Python: https://www.python.org/downloads/
* Git: https://git-scm.com/downloads

1- clone the repository and project setup.
Run the following comments on your terminal
* git clone https://github.com/dlouima/dlouima-twitter_sentiment_analyzer.git
* cd dlouima-twitter_sentiment_analyzer
* python -m venv .venv
* .venv\Scripts\activate (Windows)
* source .venv/bin/activate (MacOs/Linux)
* pip install -r requirements.txt
* python -m nltk.downloader stopwords

2- Run and access the application
 * streamlit run main.py
 * http://localhost:8501 (to access the application)

Alternatively, you can access the online here: https://dlouima-twittersentimentanalyzer-app.streamlit.app/
the app may take some time to load the first time
