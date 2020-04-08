import ast
import math
from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import logging
import pandas as pd


# Load Sentence model (based on BERT) from local
model = SentenceTransformer('bert-base-nli-mean-tokens')


def square_rooted(x):
    return math.sqrt(sum([a * a for a in x]))


def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return numerator / float(denominator)


df = pd.read_csv("/home/rathorology/PycharmProjects/Text-summarizer/capterra/salesforce.csv")
df = df.dropna(axis=0)
list_rew = list()
for index, row in df.iterrows():
    lit = ast.literal_eval(row['body'])
    combining = ""
    for k in lit.keys():
        combining += str(lit[k])
    list_rew.append(combining)

df['body'] = list_rew
# df.to_csv('/home/rathorology/PycharmProjects/Text-summarizer/capterra/salesforce_extracted.csv',index=False)

category_list = df.columns.tolist()
category_list.remove('body')
category_list.remove('_id')
category_list.remove('name')

category_desc_dict = dict()
for c in category_list:
    for i in ast.literal_eval(df[c][0]):
        df[str(c) + ":" + i['title']] = ""
        category_desc_dict[str(c) + ":" + i['title']] = i['description']
print(category_desc_dict)
df = df.drop(category_list, axis=1)
feature_vectors = dict()
for key, value in category_desc_dict.items():
    one_feature_vector = model.encode(value)
    feature_vectors[key] = one_feature_vector[0]

new_frame = pd.DataFrame()
for index, row in df.iterrows():
    print(index)
    dummy_sentance = row['body']
    review_vector = model.encode(dummy_sentance)[0]
    for key, value in feature_vectors.items():
        similarity = cosine_similarity(review_vector, value)
        row[key] = similarity
    new_frame = new_frame.append(row, ignore_index=True)
new_frame.to_csv("bert_results.csv", index=False)
