# autoemoji

This project can be used as a baseline for automated reaction sending in Telegram.
I extracted messages from one of my Telegram accounts, with total amount of 8k messages.
Almost all of them are in Russian, that is why DeepPavlov's RuBERT model was used.


### First, we'll extract messages from all groups of an Telegram account.
Pyrogram python library will be used, and Telegram app's credentials are necessary.


### With file `data.py` you can extract all messages from groups with at least 1 reaction.
It is assumed that your Telegram app's credentials are in `tg_credentials` file at first 2 lines.


### `to_csv.py` file will transform data to `pandas.Dataframe`
columns are:
- text, for message's text
- total, for message's total amount of reactions
- column for each reaction with ratio `reaction count / total`
- label for message's most popular emoji


### `get_embeddings.ipynb` notebook 
can be used for getting embeddings with RuBERT or any other transformer.
At current stage amount of tokens is limited to 128.
However, on the set of 8k messages file with binary embeddings is still about 3,3GB.
`train.ipynb` notebook contains some approaches for model training using embeddings from previous notebook.
Model training will take a lot of time because of params number. 
`LGBM_model` and `LGBM_model_default` files contain binaries of trained models for this approach.
Training in this case takes about 2,5 hours in total.

### `train.py` file
can be used for text classification task on prepared dataset using any transformer.

### `linear_approach.ipynb` notebook 
contains linear approach for problem using
TfidfVectorizer from nltk + LightGBM Classifier.
2 models are trained :
- first is reaction predicting itself (`LGBM_model_vectors`)
- second is for choosing whether to post reaction or not, default reaction is like, and model determines whether we should post like or not. (`LGBM_model_vectors_default`)
For both models we use LightGBM Classifier. Both models are trained on my dataset.


###  With script `inference.py` you can run the bot
`tg_credentials` file is necessary again. Script will ask whether you want to use RuBERT or not, 
then, if message is from chat specified in `group_ids` list, reaction will be sent. `group_ids` must be stored in file `groups`, group id for every line.
