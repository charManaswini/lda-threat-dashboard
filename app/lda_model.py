from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis.gensim_models as gensimvis

def train_lda_model(tokenized_docs, num_topics=5):
    dictionary = corpora.Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(text) for text in tokenized_docs]
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10, random_state=42)
    return lda_model, corpus, dictionary

def get_lda_viz(lda_model, corpus, dictionary):
    return gensimvis.prepare(lda_model, corpus, dictionary)
