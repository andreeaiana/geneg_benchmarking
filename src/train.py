# train.py

# import libraries
import pickle
import argparse
import re
import pandas as pd
import numpy as np

# import custom code
from src.config import MODELS_DIR, DataFrameColumns, FILENAME_TRAIN_RATINGS
from src.models import MODELS, TemplateContentRecommender
from src.fetch_data import get_raw_articles
from src.util.logger import setup_logging


def remove_author_line(*, idx: int):

    # get author and text
    author = data.loc[idx, DataFrameColumns.author]
    text = data.loc[idx, DataFrameColumns.content]

    # account for NaN values
    author = author if not pd.isna(author) else None

    # infer author name if not given
    if '<p>von' in text.lower():

        pattern_start = '<p>Von '
        pattern = f'{pattern_start}(.*?)\s*\.*(\||\. Der Chef| \(seit zwei Monaten auf Lesbos\) \||<\/p>)'

        first_paragraph = text.replace('\xa0', ' ')[:200]

        result = re.findall(pattern=pattern, string=first_paragraph,
                            flags=re.IGNORECASE)  # enforce beginning of text to avoid matching pattern later in text

        # if pattern exist
        if result:

            if result[0][1] == '|':
                string2replace = f'{pattern_start}{result[0][0]} {result[0][1]}'
            else:
                string2replace = f'{pattern_start}{result[0][0]}{result[0][1]}'

            # if first paragraph contains more than just author, add paragraph symbol again
            if result[0][1] != '</p>':
                text = re.sub(re.escape(string2replace), '<p>', text, flags=re.IGNORECASE)

            # account for special occasions where author is followed by a dot
            author_string = f'Von {author}.'
            string2replace = author_string if author_string in text else string2replace

            text = re.sub(re.escape(string2replace), '', text, flags=re.IGNORECASE)

            # remove empty paragraphs
            text = text.replace('<p></p>', '')

            # remove space after opening paragraph
            text = text.replace('<p> ', '<p>')

    return text


def train_model(data, model_name):

    logger.info(f'Training {model_name} model')
    model = MODELS[model_name]
    
    if model.__class__.__base__ == TemplateContentRecommender:
        
        # fit model
        model.fit(X=data[DataFrameColumns.content], y=data[DataFrameColumns.title])
    
    else:
        
        # get data
        X_train =  np.load(str(FILENAME_TRAIN_RATINGS))
        
        # fit model
        model.fit(X=X_train, y=data[DataFrameColumns.title])

    logger.info(f'Saving {model_name} model')
    filename = MODELS_DIR / (model_name + '.p')

    with open(filename, 'wb') as file:
        pickle.dump(model, file=file)


if __name__ == '__main__':
    
    # define logger
    logger = setup_logging(name=__file__, log_level='info')

    # define parser
    parser = argparse.ArgumentParser(description='train recommender')
    parser.add_argument('-m', '--model', default=None, type=str, help='type of recommender')

    # get args
    args = parser.parse_args()
    model_name = args.model
    
    # get data
    data = get_raw_articles()

    # remove author line prior to training
    data[DataFrameColumns.content] = [remove_author_line(idx=idx) for idx in data.index]


    # if explicit model selected train only on this model
    if model_name:

        train_model(data=data, model_name=model_name)

    # train on all
    else:

        # train each model
        for model_name in MODELS.keys():

            logger.info(f'Training with model {model_name}')
            train_model(data=data, model_name=model_name)
