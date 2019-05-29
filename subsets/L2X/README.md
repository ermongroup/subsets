# Model explanations

The code in this directory is adapted from `https://github.com/Jianbo-Lab/L2X`.

- `imdb_word` contains code for explaining a word-level CNN model on IMDB reviews.
- `imdb_sent` contains code for explaining a Hierarchical LSTM model on IMDB reviews.

The main files in each are `explain.py`.

To download the data for `imdb_sent`:
Download IMDB data from `https://www.kaggle.com/c/word2vec-nlp-tutorial/data`
Put `labeledTrainData.tsv` and `testData.tsv` in `subsets/L2X/imdb-sent/data/`
