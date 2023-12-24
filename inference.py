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

telegram_data = open("tg_credentials").readlines()
api_id = int(telegram_data[0])
api_hash = telegram_data[1][:-1]
uvloop.install()
use_rubert = input("Do you want to use rubert? Type 'y' in this case.\n") == "y"
group_ids = [int(group_id) for group_id in open("groups").readlines()]
if use_rubert:
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    model_name = "DeepPavlov/rubert-base-cased"
    model = transformers.AutoModel.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name
    )
    model.to(device)
    predictor = pickle.load(open("models/LGBM_model", "rb"))
    default_predictor = pickle.load(open("models/LGBM_model_default", "rb"))
else:
    vectorizer = pickle.load(open("models/vectorizer", "rb"))
    predictor = pickle.load(open("models/LGBM_model_vectors", "rb"))
    default_predictor = pickle.load(open("models/LGBM_model_vectors_default", "rb"))
with open('emoji_list') as emoji_file:
    reaction_list = pickle.load(emoji_file)
app = pyrogram.Client(api_id=api_id, api_hash=api_hash, name="inference")


@app.on_message()
async def scan_message(client, message):
    if use_rubert:
        inputs = tokenizer(
            message.text,
            return_tensors="pt",
            pad_to_max_length=True,
            max_length=128,
            truncation=True,
        ).to(device)
        out = model(**inputs)[0]
        out_np = out.detach().cpu().numpy().reshape((1, -1))
        reaction = predictor.predict(out_np)
        is_default = default_predictor.predict(out_np)
    else:
        message_vector = vectorizer.transform([message.text])
        reaction = predictor.predict(message_vector)
        is_default = default_predictor.predict(message_vector)
    if (reaction[0] != reaction_list[0] or not is_default[0]) and message.chat.id in group_ids:
        await app.send_reaction(message.chat.id, message.id, reaction[0])


app.run()
