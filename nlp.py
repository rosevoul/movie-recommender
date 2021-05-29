# %%
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

def init_nltk():
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('stopwords')

def nlp_prep(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    VERB_CODES = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}

    text = text.lower()
    temp_plot = []
    words = nltk.word_tokenize(text)
    tags = nltk.pos_tag(words)
    for i, word in enumerate(words):
        if tags[i][1] in VERB_CODES:
            lemmatized = lemmatizer.lemmatize(word, 'v')
        else:
            lemmatized = lemmatizer.lemmatize(word)
        if lemmatized not in stop_words and lemmatized.isalpha():
            temp_plot.append(lemmatized)

    plot = ' '.join(temp_plot)
    plot = plot.replace("n't", " not")
    plot = plot.replace("'m", " am")
    plot = plot.replace("'s", " is")
    plot = plot.replace("'re", " are")
    plot = plot.replace("'ll", " will")
    plot = plot.replace("'ve", " have")
    plot = plot.replace("'d", " would")

    return plot


# Test fuction
if __name__ == "__main__":
    # init_nltk() 
    movie_plots = pd.read_csv('data/movie_plots.csv', index_col='title')

    movie_plots["plot_nlp"] = movie_plots["plot"].apply(nlp_prep)
    movie_plots.head()
