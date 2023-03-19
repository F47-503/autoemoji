import pandas as pd
import numpy as np
import pickle
data = pickle.load(open('data_raw', 'rb'))
emoji_stats = {}
for entry in data:
  emoji_list = entry[1]
  for emoji in emoji_list:
    if not emoji.emoji in emoji_stats:
      emoji_stats[emoji.emoji] = 0
    emoji_stats[emoji.emoji] += emoji.count
df = pd.DataFrame([[entry[0], sum([x.count for x in entry[1]])] for entry in data], columns=['text', 'total'])
for emoji in emoji_stats:
  df[emoji] = pd.Series(np.zeros(df.shape[0]))
for entry in df.index:
  emoji_list = data[entry][1]
  for emoji in emoji_list:
    df[emoji.emoji][entry] += emoji.count / df['total'][entry]
df.to_csv('data_all.csv', index=False)
