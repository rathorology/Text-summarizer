import ast
import os
import time
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.losses import cosine_similarity
import pandas as pd
import numpy as np
import logging
import multiprocessing as mp

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
############################################################################################################################
df = pd.read_csv("/home/rathorology/PycharmProjects/Text-summarizer/capterra/crm.csv", engine='python')

df = df.drop(['url', 'desc'], axis=1)
df = df.dropna(axis=0)
category_list = ast.literal_eval(df['category'][0])

# remove punctuation marks
# punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'

category_desc_dict = dict()
for c in category_list:
    category = c['category']
    for i in c['feature_list']:
        # creating columns
        df[str(category) + ":" + i['title']] = ""
        # # text preprocessing
        # t = ''.join(ch for ch in i['description'] if ch not in set(punctuation))
        # # t = t.lower()
        # t = t.replace("[0-9]", " ")
        # t = ' '.join(t.split())
        category_desc_dict[str(category) + ":-" + i['title']] = i['description']
# print(category_desc_dict)
list_of_sent = list()
for i in df['body']:
    sent = list(ast.literal_eval(i).values())
    sent = ' '.join(sent)
    # preprocessing review
    # sent = ''.join(ch for ch in sent if ch not in set(punctuation))
    # sent = sent.lower()
    # sent = sent.replace("[0-9]", " ")
    # sent = ' '.join(sent.split())
    list_of_sent.append(sent)
df['review'] = list_of_sent

## import ELMO
elmo = hub.load('/home/rathorology/PycharmProjects/Text-summarizer/3')

feature_vectors = dict()
for key, value in category_desc_dict.items():
    test_input = tf.constant([value], dtype=tf.string)
    embeddings = elmo.signatures['default'](test_input)["elmo"]
    feature_vectors[key] = tf.math.reduce_mean(tf.constant(embeddings), 1)

del category_desc_dict, category_list, list_of_sent


def elmo_vectors(x):
    global i
    i += 1
    print("Row count = ", i, "Time =", round(time.time(), 2))
    print("=================================================")
    # with splitting by (.)
    splitted_review = (x.split("."))
    test_input = tf.constant(splitted_review, dtype=tf.string)
    embeddings = elmo.signatures['default'](test_input)["elmo"]

    # average out all sentance to one
    embeddings = tf.math.reduce_mean(embeddings, axis=(0, 1))

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.tables_initializer())
        return sess.run(tf.constant(embeddings))


i = 0
review_vector_frame = df[['_id', 'name', 'category_id', 'category', 'review']]
del df
# review_vector_frame = review_vector_frame[0:10]


# Parallel Processing Method
p = mp.Pool(mp.cpu_count())
review_vector_frame['review_vector'] = p.map(elmo_vectors, review_vector_frame['review'])

# #Apply Method
# review_vector_frame['review_vector'] = review_vector_frame['review'].apply(elmo_vectors)


# Cosine Similarity
for key, value in feature_vectors.items():
    v = [ele for ele in value.numpy() for i in range(review_vector_frame.shape[0])]
    similarity = np.abs(np.array(cosine_similarity(review_vector_frame['review_vector'].tolist(), v))) * 100
    review_vector_frame[key] = similarity
review_vector_frame = review_vector_frame.drop(['category', 'review_vector'], axis=1)
review_vector_frame.to_csv("results/CRM_categorization.csv", index=False)
