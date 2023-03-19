import pandas as pd
import numpy as np
import pickle
import tqdm
from sentence_transformers import SentenceTransformer, util
data = pd.read_csv('data_all.csv')
#will use deepPavlov for now
model = SentenceTransformer('DeepPavlov/rubert-base-cased-sentence')
embeddings = model.encode(data['text'])
f = open('embeddings', 'wb')
pickle.dump(embeddings, f)
f.close()
