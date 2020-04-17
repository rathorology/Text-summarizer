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
            dummy_frame['category'] = str(one_line['features'])
            # for i in one_line['features']:
            #     dummy_frame[i['category']] = str(i['feature_list'])

            dummy_frame['_id'] = one_line['_id']['$oid']
            dummy_frame['name'] = one_line['name']
            # dummy_frame['url'] = one_line['url']
            # dummy_frame['desc'] = one_line['desc']
            dummy_frame['category_id'] = one_line['category_id'][0]['$oid']
            full_frame = full_frame.append(dummy_frame)
            print('Count {} | Shape "{}"'.format(count, full_frame.shape))
        except Exception as e:
            pass

full_frame.to_csv('/home/rathorology/PycharmProjects/Text-summarizer/capterra/g2_all.csv', index=False)

#####################################################################################################################################

# filename = "/home/rathorology/PycharmProjects/Text-summarizer/capterra/capterra.json"
# with open(filename, 'r') as f:
#     objects = list(ijson.items(f, 'details.item'))
#     columns = list(objects)
# f.close()
# detail = pd.read_csv("/home/rathorology/PycharmProjects/Text-summarizer/capterra/features.csv")
# error = 0
# frame = pd.DataFrame()
# for c, d in zip(columns, detail.values):
#     df = pd.DataFrame(c)
# try:
#     df['Category'] = d[0]
#     df['Features'] = d[1]
#
#     df = df[['prosText', 'vendorId', 'title',
#              'productName', 'generalComments',
#              'productId', 'consText',
#              'reviewId', 'Category',
#              'Features']]
#     frame = frame.append(df)
# except Exception as e:
#     error += 1
#     pass
# print(frame.shape[0])
#
# print(error)
# frame.to_csv('/home/rathorology/PycharmProjects/Text-summarizer/capterra/all.csv', index=False)
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
## Category and their product with reviews count
# import pandas as pd
#
# df = pd.read_csv("capterra/g2_all.csv")
# c = df['category_id'].unique()
# # print("shape = ", df.shape)
# #
# # print('Total categories = ', len(c))
# #
# # print('Total Products = ', len(df['name'].unique()))
#
# counts = df['category_id'].value_counts()
#
# # print(counts)
# top_20_categories = counts.index.tolist()
# frame = pd.DataFrame()
# for cat in top_20_categories:
#     sample = df[df['category_id'] == cat]
#     prod = sample['name'].value_counts()
#     s = pd.DataFrame()
#     s['product_name'] = prod.index
#     s['review_count'] = prod.values
#     s['category_id'] = cat
#     frame = frame.append(s)
#
# frame.to_csv('All_categories.csv', index=False)
############################################################################################################
### Chunk preprocessing code

# df_chunk = pd.read_csv("/home/rathorology/PycharmProjects/Text-summarizer/capterra/g2_all.csv", engine='python',
#                        chunksize=50000)
# chunk_list = []  # append each chunk df here
#
# batch_count = 0
# # Each chunk is in df format
# for chunk in df_chunk:
#     batch_count += 1
#     # perform data filtering
#     # CRM ID = '5e16d981f4917a64dcb59480'
#
#     chunk_filter = chunk[chunk['category_id'] == '5e16d981f4917a64dcb59480']
#
#     # Once the data filtering is done, append the chunk to list
#     chunk_list.append(chunk_filter)
#
# # concat the list into dataframe
# df_concat = pd.concat(chunk_list)
# df_concat.to_csv("/home/rathorology/PycharmProjects/Text-summarizer/capterra/crm.csv", index=False)
# print(df_concat.shape)
# exit()
