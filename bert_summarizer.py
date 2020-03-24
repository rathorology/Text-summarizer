import ast
#
# import pandas as pd
#
# df = pd.read_csv('/home/rathorology/PycharmProjects/Text-summarizer/capterra/all.csv')
# df["review"] = df["title"] + "." + df["generalComments"] + "." + df["prosText"] + "." + df["consText"]
# df = df.drop(['prosText', 'generalComments', 'consText', 'title'], axis=1)
# category_wise_list = df['Category'].value_counts()
# df = df[df['Category'] == 'Financial Risk Management']
# df = df.dropna(axis=0)
# f = list()
# for i in df['Features']:
#     a = pd.DataFrame(ast.literal_eval(i))['name'].tolist()
#     f.append(a)
# df['Features'] = f
# df.to_csv("finanical_risk_management.csv",index=False)
########################################################################################################################
import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf
import bert
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
FullTokenizer = bert.bert_tokenization.FullTokenizer
from tensorflow.keras.models import Model  # Keras is the new high level API for TensorFlow
import math

max_seq_length = 200  # Your choice here.
input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                   name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                    name="segment_ids")
bert_layer = hub.KerasLayer("/home/rathorology/PycharmProjects/Text-summarizer/1",
                            trainable=True)
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])


def get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
    return input_ids


vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)

df = pd.read_csv("/home/rathorology/PycharmProjects/Text-summarizer/finanical_risk_management.csv")
dummy_sentance = df['review'][0]

stokens = tokenizer.tokenize(dummy_sentance)
stokens = ["[CLS]"] + stokens + ["[SEP]"]

input_ids = get_ids(stokens, tokenizer, max_seq_length)
input_masks = get_masks(stokens, max_seq_length)
input_segments = get_segments(stokens, max_seq_length)
input_ids = np.array(input_ids)
input_masks = np.array(input_masks)
input_segments = np.array(input_segments)
pool_embs, all_embs = model.predict([[input_ids], [input_masks], [input_segments]])


def square_rooted(x):
    return math.sqrt(sum([a * a for a in x]))


def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return numerator / float(denominator)


for dummy_feature in ast.literal_eval(df['Features'][0]):
    f_stokens = tokenizer.tokenize(dummy_feature)
    feat_stokens = ["[CLS]"] + f_stokens + ["[SEP]"]

    f_input_ids = get_ids(feat_stokens, tokenizer, max_seq_length)
    f_input_masks = get_masks(feat_stokens, max_seq_length)
    f_input_segments = get_segments(feat_stokens, max_seq_length)

    f_input_ids = np.array(f_input_ids)
    f_input_masks = np.array(f_input_masks)
    f_input_segments = np.array(f_input_segments)
    f_pool_embs, f_all_embs = model.predict([[f_input_ids], [f_input_masks], [f_input_segments]])

    print(str(dummy_feature), cosine_similarity(all_embs[0][0], f_all_embs[0][0]))
