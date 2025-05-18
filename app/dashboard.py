import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from lda_model import train_lda_model, get_lda_viz
import pyLDAvis
import streamlit.components.v1 as components
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.utils import simple_preprocess
import nltk

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

st.set_page_config(layout="wide", page_title="Cyber Threat LDA", page_icon="🛡️")
st.title("🛡️ Cyber Threat Intelligence - Raw Text to Topics")

# File upload
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV with 'text' column", type=["csv"])

# Preprocessing function
def preprocess(doc):
    doc = re.sub(r'\s+', ' ', str(doc))
    return ' '.join([w for w in simple_preprocess(doc) if w not in stop_words])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except Exception as e:
        st.error(f"❌ Failed to read uploaded file: {e}")
        st.stop()
else:
    st.stop()

# Validate column
if 'text' not in df.columns:
    st.error("❌ CSV must have a 'text' column.")
    st.stop()

# Sample selector
sample_size = st.sidebar.selectbox(
    "📉 Number of rows to analyze",
    [10, 100, 500, 1000, 5000, len(df)],
    index=1
)
df = df.head(sample_size)

# Preprocess on the fly
with st.spinner("Preprocessing..."):
    df['processed_text'] = df['text'].apply(preprocess)

docs = df['processed_text'].dropna().astype(str).tolist()
tokenized_docs = [doc.split() for doc in docs]

if st.sidebar.checkbox("🔍 Show preprocessed sample"):
    st.write(df[['text', 'processed_text']].head(5))

# LDA Settings
num_topics = st.sidebar.slider("🎯 Number of Topics", 2, 10, 5)
lda_model, corpus, dictionary = train_lda_model(tokenized_docs, num_topics=num_topics)
topics = lda_model.show_topics(num_topics=num_topics, formatted=False)

# Show topics
st.markdown("### 📌 Extracted Topics")
for idx, topic in topics:
    words = ", ".join([word for word, _ in topic])
    st.markdown(f"**Topic {idx + 1}**: {words}")

# WordClouds
st.markdown("### 🌐 WordClouds")
cols = st.columns(num_topics)
for i, (idx, topic) in enumerate(topics):
    with cols[i % len(cols)]:
        if topic:
            wc = WordCloud(width=400, height=300, background_color="white")
            wc.generate_from_frequencies(dict(topic))
            st.image(wc.to_array(), caption=f"Topic {idx + 1}", use_container_width=True)

# pyLDAvis
if st.checkbox("🧠 Show Interactive pyLDAvis (slow)"):
    try:
        with st.spinner("Generating pyLDAvis..."):
            vis_html = pyLDAvis.prepared_data_to_html(get_lda_viz(lda_model, corpus, dictionary))
            components.html(vis_html, height=800, scrolling=True)
    except Exception as e:
        st.error(f"❌ pyLDAvis failed: {e}")
