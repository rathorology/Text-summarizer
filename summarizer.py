import ast

import pandas as pd

df = pd.read_csv('/home/rathorology/PycharmProjects/Text-summarizer/capterra/all.csv')
df["review"] = df["title"] + "." + df["generalComments"] + "." + df["prosText"] + "." + df["consText"]
df = df.drop(['prosText', 'generalComments', 'consText', 'title'], axis=1)
category_wise_list = df['Category'].value_counts()
df = df[df['Category'] == 'Financial Risk Management']
df = df.dropna(axis=0)
f = list()
for i in df['Features']:
    a = pd.DataFrame(ast.literal_eval(i))['name'].tolist()
    f.append(a)
df['Features'] = f
print(df.head())
