import streamlit as st
import nltk
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PyPDF2 import PdfReader

# Download once
nltk.download('opinion_lexicon')
from nltk.corpus import opinion_lexicon

positive_words = set(opinion_lexicon.positive())
negative_words = set(opinion_lexicon.negative())

st.title("🧠 NLP Word Cloud with Sentiment Analysis")

# -------- INPUT --------
option = st.radio("Choose Input Type:", ["Enter Text", "Upload PDF"])

text_data = ""

if option == "Enter Text":
    text_data = st.text_area("Enter your text here:")

elif option == "Upload PDF":
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file is not None:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
        text_data = text
        st.success("PDF text extracted successfully!")

# -------- PROCESSING --------

def preprocess(text):
    return re.findall(r'\b\w+\b', text.lower())

if text_data:
    words = preprocess(text_data)

    # Sentiment classification
    pos = [w for w in words if w in positive_words]
    neg = [w for w in words if w in negative_words]

    # -------- BUTTONS --------

    if st.button("Show Sentiment Analysis"):
        st.subheader("📊 Sentiment Analysis")
        st.write("Positive Words:", pos)
        st.write("Negative Words:", neg)
        st.write("Positive Count:", len(pos))
        st.write("Negative Count:", len(neg))

    if st.button("Generate Word Cloud"):

        def color_func(word, *args, **kwargs):
            if word in positive_words:
                return "green"
            elif word in negative_words:
                return "red"
            else:
                return "gray"

        wc = WordCloud(width=800, height=400, background_color='white') \
                .generate(" ".join(words))

        fig, ax = plt.subplots()
        ax.imshow(wc.recolor(color_func=color_func))
        ax.axis("off")

        st.pyplot(fig)

else:
    st.info("Please enter text or upload a PDF.")