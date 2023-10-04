import pandas as pd
import numpy as np
import pickle

data = pickle.load(open("data_raw", "rb"))
emoji_stats = {}
for entry in data:
    emoji_list = entry[1]
    for emoji in emoji_list:
        if not emoji.emoji in emoji_stats:
            emoji_stats[emoji.emoji] = 0
        emoji_stats[emoji.emoji] += emoji.count
df = pd.DataFrame(
    [[entry[0], sum([x.count for x in entry[1]]), entry[2]] for entry in data],
    columns=["text", "total", "chat_type"],
)
for emoji in emoji_stats:
    df[emoji] = pd.Series(np.zeros(df.shape[0]))
for entry in df.index:
    emoji_list = data[entry][1]
    for emoji in emoji_list:
        df[emoji.emoji][entry] += emoji.count / df["total"][entry]
df["label"] = pd.Series(
    map(
        lambda x: df.columns[3:][x],
        np.argmax(df.loc[:, df.columns[3:]].to_numpy(), axis=1),
    )
)
df = df[~df['label'].isna()]
emoji_counts = df["label"].value_counts().head()
poor_presented_emojis = pd.Series(
    [emoji for emoji in emoji_counts.index if emoji_counts[emoji] < 200]
)
df = df[~df["label"].isin(poor_presented_emojis)]
df.drop(poor_presented_emojis, axis=1, inplace=True)
emoji_counts = emoji_counts[~emoji_counts.index.isin(poor_presented_emojis)]
df.to_csv("data_all.csv", index=False)
with open('emoji_list', 'wb') as emoji_file:
    pickle.dump(emoji_counts.index, emoji_file)
