# preprocess.py

# import libraries
import pandas as pd
import string
from tqdm import tqdm
import re
from pathlib import Path
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.cistem import Cistem
from bs4 import BeautifulSoup

# import custom code
from src.config import STOP_WORDS


def remove_links(text):

    # add empty character between two paragraph html tags to make processing easier
    text = re.sub('</p><p>', '</p> <p>', text)

    #return re.sub('http[s]*[:]*[\/]*[a-z0-9./-]*', '', text, flags=re.MULTILINE)
    return re.sub(r"http\S+", "", text)


def remove_html_tags(text):

    # prevent missing white space in further deletion of html symbols
    text = re.sub('<p>', ' ', text)

    # add punctuation after headings
    text = re.sub('</h1>', '. ', text)
    text = re.sub('</h2>', '. ', text)

    soup = BeautifulSoup(text, features="html.parser")
    text = soup.get_text()

    return text


def seperate_numbers_from_text(text):

    regex = re.finditer(r'\d+', text)

    for idx, number in enumerate(regex):
        idx_start, idx_end = number.span()

        # correct for manually introduced spaces
        idx_start = idx_start + 2 * idx
        idx_end = idx_end + 2 * idx

        text = text[:idx_start] + ' ' + text[idx_start:idx_end] + ' ' + text[idx_end:]

    return text


def remove_elements_from_text(text: str, elements: str) -> str:

    # define translation table to remove symbols
    translation = str.maketrans("", "", elements)

    # apply translation table
    text = text.translate(translation)

    return text


def remove_punctuation_from_text(text: str) -> str:
    return remove_elements_from_text(text=text, elements=string.punctuation)


def remove_special_symbols(text: str) -> str:

    # replace symbols with space
    for sym in ['-', '–']:
        text = text.replace(sym, ' ')

    # remove symbols
    text = remove_elements_from_text(text=text, elements="„“‘«»”‚’●©²…—§►•°›‹…▶´\N{SOFT HYPHEN}")

    return text


def remove_stop_words(words: list, stop_words=STOP_WORDS) -> str:

    text_without_stopwords = " ".join([word for word in words if word not in stop_words])

    return text_without_stopwords


def preprocess(text: str,
               bool_to_lowercase=True,
               bool_remove_html_tags=True,
               bool_remove_links=True,
               bool_remove_special_symbols=True,
               bool_remove_punctuation=True,
               bool_seperate_numbers_from_text=True,
               bool_stemming=True,
               bool_word_tokenize=True,
               bool_sentence_tokenize=False
               ) -> str:

    # todo: lemmatization

    # transform to lower case
    if bool_to_lowercase:
        text = text.lower()

    # remove symbols, html-tags and links
    if bool_remove_html_tags:
        text = remove_html_tags(text)

    if bool_remove_links:
        text = remove_links(text)

    if bool_remove_special_symbols:
        text = remove_special_symbols(text)

    # separate numbers from text
    if bool_seperate_numbers_from_text:
        text = seperate_numbers_from_text(text)
        
    if bool_remove_punctuation:
        text = remove_punctuation_from_text(text)

    # tokenize words
    if bool_word_tokenize:

        text_list = word_tokenize(text)

        if bool_stemming:
            text_list = [Cistem(case_insensitive=False).stem(token) for token in text_list]

        # remove stopwords
        text = remove_stop_words(text_list)

    if bool_sentence_tokenize:
        text = sent_tokenize(text, language='german')

    return text

if __name__ == '__main__':
    pass