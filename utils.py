import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# load stopwords
try:
    stop_words = stopwords.words('english')
except:
    nltk.download('stopwords')
    stop_words = stopwords.words('english')

stemmer = PorterStemmer()

def clean_text(text):
    text = str(text).lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [stemmer.stem(w) for w in words]  # stemming added
    return " ".join(words)