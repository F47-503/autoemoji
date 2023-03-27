import asyncio
import pyrogram
import uvloop
import tgcrypto
import pandas as pd
import pickle
import transformers
import torch
import warnings
import numpy as np
from nltk.stem.porter import *
warnings.filterwarnings("ignore")

telegram_data = open('tg_credentials').readlines()
api_id = int(telegram_data[0])
stemmer = PorterStemmer()
api_hash = telegram_data[1][:-1]
uvloop.install()
use_rubert = input("Do you want to use rubert? Type 'y' in this case.\n") == 'y'
group_ids = [int(group_id) for group_id in open('groups').readlines()]
if use_rubert:
  device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
  model = transformers.AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")
  tokenizer = transformers.AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
  model.to(device)
  predictor = pickle.load(open('models/XGB_model', 'rb'))
  default_predictor = pickle.load(open('models/XGB_model_default', 'rb'))
else:
  vectorizer = pickle.load(open('models/vectorizer', 'rb'))
  predictor = pickle.load(open('models/XGB_model_vectors', 'rb'))
  default_predictor = pickle.load(open('models/XGB_model_vectors_default', 'rb'))
reaction_list = [
    '👍',
    '\U0001fae1',
    '🙏',
    '🔥',
    '🥰',
    '👌',
    '🤓',
    '🤝',
    '👎',
    '🤬',
    '😁',
    '🤡',
    '🏆',
    '👏',
    '💯',
    '❤',
    '🌚',
    '🤨',
    '🥴',
    '🤩',
    '😍',
    '🤣',
    '😢',
    '💩',
    '🤯',
    '❤\u200d🔥',
    '🐳',
    '🤮',
    '🤗',
    '😇',
    '🤔',
    '🖕',
    '🥱',
    '😈',
    '🕊',
    '🍌',
    '🌭',
    '💋',
    '⚡',
    '🍓',
    '🍾',
    '💔',
    '😱',
    '🎉',
    '😐',
    '✍',
    '😭',
    '🆒',
    '🗿',
    '👀',
    '💅',
    '🎄',
    '☃',
    '👨\u200d💻',
    '👻',
    '🙊',
    '🤪',
    '😨',
    '💊',
    '😴',
]
app = pyrogram.Client(api_id=api_id, api_hash=api_hash, name="inference")
@app.on_message()
async def scan_message(client, message):
  if use_rubert:
    inputs = tokenizer(message.text, return_tensors="pt", pad_to_max_length=True, max_length=128, truncation=True).to(device) 
    out = model(**inputs)[0]
    out_np = out.detach().cpu().numpy().reshape((1,-1))
    reaction = predictor.predict(out_np) 
    is_default = default_predictor.predict(out_np)
  else:
    text = ' '.join([stemmer.stem(word) for word in message.text.split(' ')])
    message_vector = vectorizer.transform([text])
    reaction = np.argmax(predictor.predict(message_vector), axis=1) 
    is_default = default_predictor.predict(message_vector)
  if message.chat.id not in group_ids:
    print(message.chat.id)
  if (reaction[0] != 0 or not is_default[0]) and message.chat.id in group_ids:
    await app.send_reaction(message.chat.id, message.id, reaction_list[reaction[0]])
app.run()


