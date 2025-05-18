import re
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess

stop_words = set(stopwords.words("english"))

def clean_and_tokenize(doc):
    doc = re.sub(r"\s+", " ", str(doc))
    return [token for token in simple_preprocess(doc, deacc=True) if token not in stop_words]
