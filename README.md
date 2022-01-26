# GeNeG Benchmarking
Benchmarking experiments of different news recommender systems using GeNeG and its corresponding news corpus.

## Recommendation models
- Content-based recommnders
	- TF-IDF
	- Word2vec
	- Transformer
- Collaborative filtering recommenders
	- Alternating Least Squares (ALS)
- Knowledge-aware recommenders
	- RippleNet
	- DKN

## Usage

Configurations for directories, filepaths, and some model parameters can be set in `config.py`.

### Content-based and collaborative filtering recommendation models
Train a model
```
python -m src.train
```

Evaluate a model
```
python -m src.evaluate
```

### RippleNet
Prepare data for RippleNet
```
python -m src.preprocess_ripplenet
```

Train and evaluate RippleNet
```
python -m src.run_ripplenet
```

### DKN
Prepare data for DKN
```
python -m src.preprocess_dkn
```

Preprocess news data and train Word2vec model
```
python -m src.dkn_news_preprocess
```

Preprocess entity data and train TransX model
``` 
python -m src.prepare_data_for_transx
python -m src.transx.train_transe (note: you can also choose other KGE methods)
python -m src.dkn_kg_preprocess
```

Train and evaluate DKN
```
python -m src.run_dkn
```

## Data
The data necessary to run the models can be found in the `data` folder.

The article's content is required to train the content-based recommender systems and to compute recommendations. A sample of the news corpus is available in the  `data/articles.csv` file. Due to copyright policies, this sample does not contain the abstract and body of the articles.

A full version of the news corpus is available [upon request](mailto:andreea@informatik.uni-mannheim.de).

## Requirements
This code is implemented in Python 3. The requirements can be installed from `requirements.txt`.

```
pip3 install -r requirements.txt
```

## License
The code is licensed under the MIT License. The data and knowledge graph files are licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

Parts of the code were originally forked and adapted from:
- [RippleNet-TF2](https://github.com/tezignlab/RippleNet-TF2)
- [DKN](https://github.com/hwwang55/DKN)
- [OpenKE](https://github.com/thunlp/OpenKE)

We owe many thanks to the authors of RippleNet-TF2, DKN, and OpenKE for making their codes available.
