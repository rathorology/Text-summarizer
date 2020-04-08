import ijson

import pandas as pd
from numpy import nan as Nan

# Use it for json


filename = "/home/rathorology/PycharmProjects/Text-summarizer/capterra/g2_products 2.json"
import ijson
import json

# ['_id','name', 'desc', 'category_id', 'details', 'reviews', 'features']
full_frame = pd.DataFrame()
count = 0
with open(filename) as f:
    for line in f:
        count += 1
        one_line = json.loads(line)
        try:

            dummy_frame = pd.DataFrame()

            sample = pd.DataFrame(one_line['reviews'])
            dummy_frame['body'] = sample['body']
            for i in one_line['features']:
                dummy_frame[i['category']] = str(i['feature_list'])

            dummy_frame['_id'] = one_line['_id']['$oid']
            dummy_frame['name'] = one_line['name']
            print('Count {} | Shape "{}"'.format(count, dummy_frame.shape[1]))
            # dummy_frame['category_id'] = one_line['category_id'][0]['$oid']
            full_frame = full_frame.append(dummy_frame)
        except Exception as e:
            pass

full_frame.to_csv('/home/rathorology/PycharmProjects/Text-summarizer/capterra/g2_all.csv', index=False)

#####################################################################################################################################

filename = "/home/rathorology/PycharmProjects/Text-summarizer/capterra/capterra.json"
with open(filename, 'r') as f:
    objects = list(ijson.items(f, 'details.item'))
    columns = list(objects)
f.close()
detail = pd.read_csv("/home/rathorology/PycharmProjects/Text-summarizer/capterra/features.csv")
error = 0
frame = pd.DataFrame()
for c, d in zip(columns, detail.values):
    df = pd.DataFrame(c)
try:
    df['Category'] = d[0]
    df['Features'] = d[1]

    df = df[['prosText', 'vendorId', 'title',
             'productName', 'generalComments',
             'productId', 'consText',
             'reviewId', 'Category',
             'Features']]
    frame = frame.append(df)
except Exception as e:
    error += 1
    pass
print(frame.shape[0])

print(error)
frame.to_csv('/home/rathorology/PycharmProjects/Text-summarizer/capterra/all.csv', index=False)
#################################################################################################################################


# Use it for json

#
# filename = "/home/rathorology/PycharmProjects/Text-summarizer/capterra/capterra_products.json"
# import ijson
#
# with open(filename, 'r') as f:
#     objects = ijson.items(f, 'RECORDS.item.details')
#     columns = list(objects)
#
# # columns = columns[10000:20000]
#
# frame = pd.DataFrame()
# for c in columns:
#
#     df = pd.DataFrame(c['features'])
#     frame = frame.append(df)
#     if frame.shape[0] > 25000:
#         break
#     # df = pd.DataFrame(index=[0])
#     # frame = frame.append(df)
#
#
#
# frame.to_csv('/home/rathorology/PycharmProjects/Text-summarizer/capterra/25k_features.csv', index=False)
#################################################################################################################################

#
# import pandas as pd
# import gensim
# from nltk import RegexpTokenizer
# from nltk.corpus import stopwords
# from gensim.summarization import keywords
# from gensim.summarization import summarize
# import numpy as np
#
# df = pd.read_csv("capterra/reviews.csv")
# df = df[['globalReviewId', 'productName', 'prosText', 'generalComments', 'consText']]
#
# df["review"] = df["generalComments"] + " " + df["prosText"] + " " + df["consText"]
# df = df.drop(['prosText', 'generalComments', 'consText'], axis=1)
# df = df.dropna(axis=0)
# df = df.iloc[:300]
#
#
# def clean_sent_words(sent):
#     sent = str(sent).lower()
#     sent = RegexpTokenizer(r'\w+').tokenize(sent)
#     stop = set(stopwords.words('english'))
#     sent_words = [word for word in sent if word not in stop]
#     return sent_words
#
#
# # clean_review = list()
# # for i in df['review'].tolist():
# #     clean_review.append(clean_sent_words(i))
# # df['clean_review'] = clean_review
#
# summary_list = list()
# keyword_list = list()
# for i in df['review'].tolist():
#     keyword = keywords(i)
#     keyword_list.append(keyword)
#
#     # print('=================Keyword===============')
#     # print(keyword)
#     # print('=========================================')
#
#     try:
#         sum = summarize(i)
#         summary_list.append(sum)
#     except Exception as e:
#         summary_list.append(np.NaN)
#         print('nan')
#     # print('===================Original Review=================================')
#     # print(df['review'][1])
#     # print('==============================================================================================================')
#     # print('=================Summary==========================================')
#     # print(sum)
# df['keyword'] = keyword_list
# df['summary'] = summary_list
# df = df.dropna(axis=0)
# df.to_csv('sample_reviews_summary.csv', index=False)
#
# df = pd.read_csv("sample_reviews_summary.csv")
# print(df.isnull().sum())
# df = df.dropna(axis=0)
# df.to_csv('sample_reviews_summary.csv', index=False)
# df = pd.read_csv("sample_reviews_summary.csv")
# print(df.isnull().sum())
