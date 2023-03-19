# autoemoji

This project can be used as a baseline for automated reaction sending in Telegram.
I exctracted messages from one of my Telegram accounts, with otal amount of 8k messages.
Almost all of them are in Russian, that is why DeepPavlov's RuBERT model was used.


### First, we'll exctract messages from all groups of an Telegram account.
Pyrogram python library will be used, and Telegram app's credentials are necessary.


### With file `data.py` you can exctract all messages from groups with at least 1 reaction.
It is assumed that your Telegram app's credentials are in `tg_credentials` file at first 2 lines.


### `to_csv.py` file will transform data to `pandas.Dataframe`
columns are:
- text, for message's text
- total, for message's total amount of reactions
- column for each reaction with ratio `reaction count / total`


### `auto_sentence_transformer.py` 
exctracts embeddings from data using auto generated sentence transformer from RuBERT.


### `get_embeddings.ipynb` notebook 
can be used for getting embeddings with RuBERT.
At current stage amount of tokens is limited to 64.
However, on the set of 8k messages file with binary embeddings is still about 3,3GB.
`train.ipynb` notebook contains some approaches for model training using embeddings from previous notebook.
Model training will take a lot of time because of params number.


### `linear_approach.ipynb` notebook 
contains linear approach for problem using
TfidfVectorizer from nltk + LogisticRegression from sklearn.
LightGBM or XGBoost classifiers can also be used, but accuracy and perfomance will be lower.
2 models are trained :
- first is reaction predicting itself
- second is for choosing whether to post reaction or not, default reaction is like, and model determines whether we should post like or not.
For both models we use Logistic Regression with weighted classes.

`logreg_model_balanced` is binary for main model, and `logreg_model_default` is binary for default-determining model, both models were trained on my dataset.
