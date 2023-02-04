import string
import requests
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords


def text_process(text):
    '''
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Return the cleaned text as a list of words
    4. Remove words
    '''
    stemmer = WordNetLemmatizer()
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join([i for i in nopunc if not i.isdigit()])
    nopunc = [word.lower() for word in nopunc.split() if word not in stopwords.words('english')]
    return [stemmer.lemmatize(word) for word in nopunc]


def get_old_testament(bible_array):
    old_testament_array = []
    for a in bible_array:
        if 'Matthew ' in a[0]:
            print(a)
            break
        else:
            old_testament_array.append(a)
    return old_testament_array


def chapter_split(text):
    reference_list = text.split(' ')
    chapter_verse = reference_list[-1].split(':')
    return int(chapter_verse[0])

def verse_split(text):
    reference_list = text.split(' ')
    chapter_verse = reference_list[-1].split(':')
    return int(chapter_verse[1])

def book_split(text):
    reference_list = text.split(' ')
    chapter_verse = reference_list[0]
    return chapter_verse


def printable_string(text):
    # string.translate(text, string.printable)
    # text = text.translate(str.maketrans('', '', string.printable))
    # filtered_string = (filter(lambda x: x in string.printable, text))
    filtered_string = ''.join(s for s in text if s in string.printable)
    return filtered_string


def acquire_data(data_dir, bible_url, tfidf_fn, df_vocab_fn, df_fn):
    response = requests.get(bible_url)
    raw_bible = response.text
    bible_list = raw_bible.splitlines()
    bible_array = np.array([item.split('\t') for item in bible_list])
    bible_array[:, 1] = np.array([printable_string(verse) for verse in bible_array[:, 1]])
    df = pd.DataFrame(bible_array[3:, :], columns=['reference','text'])
    df['chapter'] = df['reference'].apply(chapter_split)
    df['verse'] = df['reference'].apply(verse_split)
    df['book'] = df['reference'].apply(book_split)
    df = df[['book', 'chapter', 'verse', 'text']]
    tfidfconvert = TfidfVectorizer(analyzer=text_process).fit(bible_array[3:,1])
    tfidf_text = tfidfconvert.transform(bible_array[3:,1])
    df_vocab = pd.DataFrame(index=None,
                            data=zip(tfidfconvert.vocabulary_.keys(),
                                     tfidfconvert.vocabulary_.values()),
                            columns=['word', 'index']).sort_values('index')
    tfidf_df = pd.DataFrame(csr_matrix.todense(tfidf_text), columns=df_vocab.word)

    tfidf_df.to_pickle(f'{data_dir}/{tfidf_fn}')
    df_vocab.to_pickle(f'{data_dir}/{df_vocab_fn}')
    df.to_pickle(f'{data_dir}/{df_fn}')
