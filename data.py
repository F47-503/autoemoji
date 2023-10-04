import asyncio
import pyrogram
import uvloop
import tgcrypto
import pandas as pd
import pickle

api_data = open("tg_credentials").read().split("\n")
api_id = int(api_data[0])
api_hash = api_data[1]
uvloop.install()
app = pyrogram.Client(api_id=api_id, api_hash=api_hash, name="data")
messages_reactions = []


def main():
    cnt = 0
    for chat in app.get_dialogs():
        if chat.chat.title:
            for mess in app.get_chat_history(chat_id=chat.chat.id):
                if mess.text and mess.reactions:
                    cnt += 1
                    if cnt % 3000 == 0:
                        print(cnt)
                    messages_reactions.append(
                        [
                            mess.text,
                            mess.reactions.reactions,
                            chat.chat.type is pyrogram.enums.ChatType.GROUP,
                        ]
                    )
            print("next chat")


app.start()
main()
with open("data_raw", "wb") as f:
    pickle.dump(messages_reactions, f)
