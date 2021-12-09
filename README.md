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


## Data
The data necessary to run the models can be found in the `data` folder.

The article's content is required to train the content-based recommender systems and to compute recommendations. A sample of the news corpus is available in the  `data/articles.csv` file. Due to copyright policies, this sample does not contain the abstract and body of the articles.

A full version of the news corpus is available [upon request](mailto:andreea@informatik.uni-mannheim.de).

## Requirements
This code is implemented in Python 3. The requirements can be installed from requirements.txt.

```
pip3 install -r requirements.txt
```

## License
Licensed under the MIT License.

Parts of the code were originally forked and adapted from:
- [RippleNet-TF2](https://github.com/tezignlab/RippleNet-TF2)

We owe many thanks to the authors of the RippleNet-TF2 for making their codes available.

<!-- ## Citation
If you use this code in your research, please cite:

```
@misc{iana2021geneg,
      title={A German News Corpus for Benchmarking Knowledge-Aware News Recommender Systems}, 
      author={Iana, Andreea and Grote, Alexander and Ludwig, Katharina and Alam, Mehwish and Müller, Phillip and Weinhardt, Christof and Paulheim, Heiko},
      year={2021}
}
``` -->
