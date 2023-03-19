import asyncio
import pyrogram
import uvloop
import tgcrypto
import pandas as pd
import pickle
f = open('tg_credentials')
api_id = int(f.read())
api_hash = f.read()
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
          messages_reactions.append([mess.text, mess.reactions.reactions])
      print('next chat')
app.start()
main()
f = open('data_raw', 'wb')
pickle.dump(messages_reactions, f)
f.close()
