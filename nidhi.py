import ijson

import pandas as pd

##################################################################################################################################
# filename = "/home/rathorology/PycharmProjects/Text-summarizer/capterra/capterra_products-002.bson"
# import bson
# with open(filename,'rb') as f:
#     data = bson.loads(f.read())
#
# print(data.keys())
# rev = data['reviews']
# df = pd.DataFrame(rev)
# df.to_csv("/home/rathorology/PycharmProjects/Text-summarizer/capterra/reviews.csv",index=False)
# print(rev)
# print(one['generalComments'])
# print('================================================================================================================================')
####################################################################################################################################


# Use it for json
# filename = "/home/rathorology/PycharmProjects/Text-summarizer/capterra/capterra_products.json"
# import ijson
#
# with open(filename, 'r') as f:
#     objects = ijson.items(f, 'RECORDS.item.reviews')
#     columns = list(objects)
#
# # df = pd.DataFrame(columns)
# # df.to_csv("/home/rathorology/PycharmProjects/Text-summarizer/capterra/dump.csv", index=False)
# frame = pd.DataFrame()
# for c in columns:
#     df = pd.DataFrame(c)
#     frame = frame.append(df)
#     print(frame.shape[0])
#     if frame.shape[0] > 100000:
#         break
#
# frame.to_csv('/home/rathorology/PycharmProjects/Text-summarizer/capterra/reviews.csv', index=False)
#################################################################################################################################


import pandas as pd
import gensim
from nltk import RegexpTokenizer
from nltk.corpus import stopwords

df = pd.read_csv("capterra/reviews.csv")
df = df[['globalReviewId', 'productName', 'prosText', 'generalComments', 'consText']]

df["review"] = df["generalComments"] + " " + df["prosText"] + " " + df["consText"]
df = df.drop(['prosText', 'generalComments', 'consText'], axis=1)
df = df.dropna(axis=0)

txt_data = list(df['review'])


# print(txt_data)


def _clean_sent_words(sent):
    sent = str(sent).lower()
    sent = RegexpTokenizer(r'\w+').tokenize(sent)
    stop = set(stopwords.words('english'))
    sent_words = [word for word in sent if word not in stop]
    return sent_words


class LabeledLineSentence(object):

    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list

    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            print(idx, doc)
            print('====================================================================')
            yield gensim.models.doc2vec.LabeledSentence(doc,
                                                        [self.labels_list[idx]])


print('txt_data = ', len(txt_data))
txt_data = _clean_sent_words(txt_data)
print('txt_data = ', len(txt_data))
# it = LabeledLineSentence(txt_data, txt_labels)


# # Prepare and Train Model
# model = gensim.models.Doc2Vec(size=300, min_count=0, alpha=0.025, min_alpha=0.025)
# model.build_vocab(it)
