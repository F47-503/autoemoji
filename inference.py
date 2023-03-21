import asyncio
import pyrogram
import uvloop
import tgcrypto
import pandas as pd
import pickle
import transformers
import torch
import warnings
warnings.filterwarnings("ignore")

telegram_data = open('tg_credentials').readlines()
api_id = int(telegram_data[0])
api_hash = telegram_data[1][:-1]
uvloop.install()
use_rubert = input("Do you want to use rubert? Type 'y' in this case.\n") == 'y'
group_ids = []
if use_rubert:
  device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
  model = transformers.AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")
  tokenizer = transformers.AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
  model.to(device)
  predictor = pickle.load(open('LGBM_model', 'rb'))
  default_predictor = pickle.load(open('LGBM_model_default', 'rb'))
else:
  vectorizer = pickle.load(open('vectorizer', 'rb'))
  predictor = pickle.load(open('LGBM_model_vectors', 'rb'))
  default_predictor = pickle.load(open('LGBM_model_vectors_default', 'rb'))
reaction_list = []
app = pyrogram.Client(api_id=api_id, api_hash=api_hash, name="inference")
@app.on_message()
def scan_message(client, message):
  print("message received")
  if use_rubert:
    inputs = tokenizer(message.text, return_tensors="pt", pad_to_max_length=True, max_length=128, truncation=True).to(device) 
    out = model(**inputs)[0]
    out_np = out.detach().cpu().numpy().reshape((1,-1))
    reaction = predictor.predict(out_np) 
    is_default = default_predictor.predict(out_np)
  else:
    message_vector = vectorizer.transform(message.text)
    reaction = predictor.predict(message_vector) 
    is_default = default_predictor.predict(message_vector)
  if (reaction != 0 or not is_default) and message.chat.id in group_ids:
    app.send_reaction(message.chat.id, message.id, reaction_list[reaction])
app.run()


