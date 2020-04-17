import ast
import json
import os
import time
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import tensorflow_hub as hub
import tensorflow as tf
import pandas as pd
import re
import numpy as np
from scipy import spatial

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
############################################################################################################################
df = pd.read_csv("/home/rathorology/PycharmProjects/Text-summarizer/capterra/crm.csv", engine='python')
df = df.drop(['url', 'desc'], axis=1)
df = df.dropna(axis=0)
category_list = ast.literal_eval(df['category'][0])


# category_list.remove('body')
# category_list.remove('_id')
# category_list.remove('name')

def sim(a, b):
    normalize_a = tf.nn.l2_normalize(a, 0)
    normalize_b = tf.nn.l2_normalize(b, 0)
    cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b))
    return cos_similarity


# remove punctuation marks
punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'

category_desc_dict = dict()
for c in category_list:
    category = c['category']
    for i in c['feature_list']:
        # creating columns
        df[str(category) + ":" + i['title']] = ""

        # text preprocessing
        t = ''.join(ch for ch in i['description'] if ch not in set(punctuation))
        t = t.lower()
        t = t.replace("[0-9]", " ")
        t = ' '.join(t.split())
        category_desc_dict[str(category) + ":-" + i['title']] = i['description']
# print(category_desc_dict)
list_of_sent = list()
for i in df['body']:
    sent = list(ast.literal_eval(i).values())
    sent = ' '.join(sent)

    # preprocessing review
    sent = ''.join(ch for ch in sent if ch not in set(punctuation))
    sent = sent.lower()
    sent = sent.replace("[0-9]", " ")
    sent = ' '.join(sent.split())

    list_of_sent.append(sent)
df['review'] = list_of_sent

# this block enables GPU enabled multiprocessing
core_config = tf.compat.v1.ConfigProto()
core_config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=core_config)
tf.compat.v1.keras.backend.set_session(session)
# ------------------------------
## import ELMO
elmo = hub.load('/home/rathorology/PycharmProjects/Text-summarizer/3')

feature_vectors = dict()
for key, value in category_desc_dict.items():
    test_input = tf.constant([value], dtype=tf.string)
    embeddings = elmo.signatures['default'](test_input)["elmo"]
    feature_vectors[key] = tf.math.reduce_mean(tf.constant(embeddings), 1)

del category_desc_dict, category_list, list_of_sent


# feature_vectors_frame = pd.DataFrame([feature_vectors])
# feature_vectors_frame.to_csv("feature_vectors_frame.csv", index=False)

def elmo_vectors(x):
    global i
    i += 1
    print("Iteration count = ", i, "Time =", round(time.time(), 2))
    # without splitting
    test_input = tf.constant(x.tolist(), dtype=tf.string)
    embeddings = elmo.signatures['default'](test_input)["elmo"]
    print("=================================================")
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.tables_initializer())
        # return average of ELMo features
        return sess.run(tf.math.reduce_mean(tf.constant(embeddings), 1))


df_batch = [df[i:i + 25] for i in range(0, df.shape[0], 25)]
i = 0

elmo_train = [elmo_vectors(x['review']) for x in df_batch]

elmo_train_new = np.concatenate(elmo_train, axis=0)
review_vector_frame = df
del df
review_vector_frame = review_vector_frame[['body', '_id', 'name', 'category_id']]
review_vector_frame['review_vector'] = elmo_train_new.tolist()
for key, value in feature_vectors.items():
    v = [ele for ele in value.numpy() for i in range(review_vector_frame.shape[0])]
    similarity = cosine_similarity(review_vector_frame['review_vector'].tolist(), v).mean(axis=1)
    review_vector_frame[key] = similarity
review_vector_frame = review_vector_frame.drop(['review_vector'], axis=1)
review_vector_frame.to_csv("CRM_categorization.csv", index=False)
