import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
for pkg in ['stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """
    Advanced text cleaning:
    - Lowercase
    - Remove URLs
    - Remove source tags like (Reuters), (AP) etc.
    - Remove punctuation & numbers
    - Remove stopwords
    - Lemmatize (better than stemming)
    """
    text = str(text).lower()

    # ✅ FIX 1: Remove source/agency tags — ye tha main culprit!
    # e.g. "(reuters)", "(ap)", "(washington)" — model inhe seekh leta tha
    text = re.sub(r'\(.*?\)', '', text)

    # ✅ FIX 2: Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # ✅ FIX 3: Remove digits
    text = re.sub(r'\d+', '', text)

    # ✅ FIX 4: Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # ✅ FIX 5: Tokenize and remove stopwords
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]

    # ✅ FIX 6: Lemmatize instead of Stem (gives real words)
    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)
