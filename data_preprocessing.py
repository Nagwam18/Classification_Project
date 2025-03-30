import pandas as pd
import re
import string
from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """Lowercase, remove punctuation, stopwords, URLs, numbers, and lemmatize."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\S+|#\w+|\d+", "", text)  # Remove URLs, numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in STOPWORDS]
    return ' '.join(words)

def load_and_preprocess_people_wiki():
    """Load and clean People Wikipedia dataset."""
    people_wiki = pd.read_csv('C:/Users/Noga/Desktop/ML2_Project/Dataset/people_wiki.csv')
    people_wiki['clean_text'] = people_wiki['text'].apply(clean_text)
    return people_wiki

def load_and_preprocess_news():
    """Load and clean 20 Newsgroups dataset."""
    categories = ['talk.religion.misc', 'comp.graphics', 'sci.space']
    news = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
    news_data = pd.DataFrame({'text': news.data, 'category': news.target})
    news_data["clean_text"] = news_data["text"].apply(clean_text)
    return news_data
