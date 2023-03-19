# autoemoji

This project can be used as a baseline for automated reaction sending in Telegram.

First, we'll exctract messages from all groups of an Telegram account.

Pyrogram python library will be used, and Telegram app's credentials are necessary.

With file `data.py` you can exctract all messages from groups with at least 1 reaction.

It is assumed that your Telegram app's credentials are in `tg_credentials` file at first 2 lines.

`to_csv.py` file will transform data to `pandas.Dataframe` with columns:
- text, for message's text
- total, for message's total amount of reactions
- column for each reaction with ratio `reaction count / total`

