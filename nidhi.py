import ijson

import pandas as pd

#
# filename = "/home/rathorology/PycharmProjects/great_learning/capterra/capterra_products-002.bson"
# import bson
# with open(filename,'rb') as f:
#     data = bson.loads(f.read())
#
# # print(data.keys())
# one = data['reviews'][0]
#
# print(one['generalComments'])


filename = "/home/rathorology/PycharmProjects/great_learning/capterra/capterra_products.json"
import ijson




with open(filename, 'r') as f:
    objects = ijson.items(f, 'RECORDS.item.reviews.item.generalComments')
    columns = list(objects)

print(columns)