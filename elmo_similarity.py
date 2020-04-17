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

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#
df = pd.read_csv("/home/rathorology/PycharmProjects/Text-summarizer/capterra/salesforce_extracted.csv", engine='python')

# # remove punctuation marks
# punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'
#
# df['body'] = df['body'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
#
# # convert text to lowercase
# df['body'] = df['body'].str.lower()
#
# # remove numbers
# df['body'] = df['body'].str.replace("[0-9]", " ")
#
# # remove whitespaces
# df['body'] = df['body'].apply(lambda x: ' '.join(x.split()))
## import spaCy's language model
# nlp = spacy.load('en_core_web_lg')
#
#
# # function to lemmatize text
# def lemmatization(texts):
#     output = []
#     for i in texts:
#         s = [token.lemma_ for token in nlp(i)]
#         output.append(' '.join(s))
#     return output
#
#
# df['body'] = lemmatization(df['body'])
# del nlp
# ------------------------------
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

elmo = hub.load('/home/rathorology/PycharmProjects/Text-summarizer/3')

feature_vectors = dict()
for key, value in category_desc_dict.items():
    test_input = tf.constant([value], dtype=tf.string)
    embeddings = elmo.signatures['default'](test_input)["elmo"]
    feature_vectors[key] = tf.math.reduce_mean(tf.constant(embeddings), 1)


# feature_vectors_frame = pd.DataFrame([feature_vectors])
# feature_vectors_frame.to_csv("feature_vectors_frame.csv", index=False)


def sim(a, b):
    normalize_a = tf.nn.l2_normalize(a, 0)
    normalize_b = tf.nn.l2_normalize(b, 0)
    cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b))
    return cos_similarity


def elmo_vectors(x):
    # with splitting by (.)
    concatinated_embadings = list()
    for one_review in x.tolist():
        splitted_review = (one_review.split("."))
        test_input = tf.constant(splitted_review, dtype=tf.string)
        embeddings = elmo.signatures['default'](test_input)["elmo"]

        # average out all sentance to one
        embeddings = tf.math.reduce_mean(embeddings, axis=(0, 1))

        concatinated_embadings.append(embeddings)
    concatinated_embadings = tf.stack(concatinated_embadings)
    global i
    i += 1
    print("Iteration count = ", i, "Time =", round(time.time(), 2))
    # # without splitting
    # test_input = tf.constant(x.tolist(), dtype=tf.string)
    # embeddings = elmo.signatures['default'](test_input)["elmo"]
    print("=================================================")
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.tables_initializer())
        # return average of ELMo features
        # return sess.run(tf.math.reduce_mean(tf.constant(concatinated_embadings), 1))
        return sess.run(tf.constant(concatinated_embadings))


df = df[0:20]
df_batch = [df[i:i + 10] for i in range(0, df.shape[0], 10)]
i = 0
# elmo_train = list()
# for x in df_batch:
#     elmo_train.append(elmo_vectors(x['body']))
elmo_train = [elmo_vectors(x['body']) for x in df_batch]

elmo_train_new = np.concatenate(elmo_train, axis=0)
review_vector_frame = df
review_vector_frame = review_vector_frame[['body', '_id', 'name']]
review_vector_frame['review_vector'] = elmo_train_new.tolist()
for key, value in feature_vectors.items():
    v = [ele for ele in value.numpy() for i in range(review_vector_frame.shape[0])]
    similarity = cosine_similarity(review_vector_frame['review_vector'].tolist(), v).mean(axis=1)
    review_vector_frame[key] = similarity
review_vector_frame = review_vector_frame.drop(['review_vector'], axis=1)
review_vector_frame.to_csv("demo.csv", index=False)
