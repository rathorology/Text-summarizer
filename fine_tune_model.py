import itertools
import random
from gensim.summarization import keywords
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
import pandas as pd
from nltk.stem import WordNetLemmatizer

df = pd.read_csv("results/elmo_results.csv")
lemmatizer = WordNetLemmatizer()
feat_dict = {
    'Sales Force Automation:Contact & Account Management': 'Store and retrieve information associated to customer contacts and accounts.  Track company-wide communication and information about contacts and accounts.',
    'Sales Force Automation:Partner Relationship Mgmt. (PRM)': 'Manage partners by tracking channel partner leads and sales opportunities.   Provide a partner portal to collaborate with channels on sales opportunities and to share product, pricing, quoting, ordering, and training information',
    'Sales Force Automation:Opportunity & Pipeline Mgmt.': 'Manage sales opportunities through their lifecycle from lead to order. Track stages, values, and probabilities of close.  Manage sales pipelines by individual sales rep, team, region, and company-wide.',
    'Sales Force Automation:Task / Activity Management': 'Manage and track tasks and activities.  Assign due dates and integrate to calendars to manage daily schedules and priorities.',
    'Sales Force Automation:Territory & Quota Management': 'Assign and manage sales quotas and territories.  Track progress against quotas.  Change as needed.',
    'Sales Force Automation:Desktop Integration': 'Allows users to sync their Email, Calendar and Contact tools with their CRM system.  Includes Microsoft Outlook and Google integration.',
    'Sales Force Automation:Product & Price List Management': 'Enter product/part numbers and manage the prices associated with them.  Typically functionality allows users to add products and prices to opportunities and quotes if these modules are provided within the same system.',
    'Sales Force Automation:Quote & Order Management': 'Allows users to create a quote to be provided to a customer that contain at least products, prices and associated discounts.  Order management allows users to process orders that contain products, prices and associated discounts.',
    'Sales Force Automation:Customer Contract Management': 'Management of contracts made with customers.  Contract management includes negotiating the terms and conditions in contracts and ensuring compliance with the terms and conditions, as well as documenting and agreeing on any changes or amendments that may arise during its implementation or execution.',
    'Marketing Automation:Email Marketing': 'Allows users to send email to contacts in bulk.  Common features include: Built in Email templates, social media integration, Subscriber list management, sign up forms, success rate reports, AB testing and auto-responders.',
    'Marketing Automation:Campaign Management': 'Optimizes the process for organizations to develop and deploy multiple-channel marketing campaigns to target groups or individuals and track the effect of those campaigns, by customer segment, over time.',
    'Marketing Automation:Lead Management': 'Allows users to manage and track leads though a process.  The lead process typically involves steps such as:  1. Lead Generation, 2. Customer Inquiry, Inquiry Capture, Lead Filtering, Lead Grading, Lead Distribution and Lead Qualification.',
    'Marketing Automation:Marketing ROI Analytics': 'Enables analysis of effectiveness of an organizations various marketing activities',
    'Customer Support:Case Management': 'Tracks issues/help requests reported by customers through the resolution process.',
    'Customer Support:Customer Support Portal': 'Provides a convenient way for customers to get answers to inquiries, post service issues, place orders, view order histories, and gain access to other information contained in the knowledge base.',
    'Customer Support:Knowledge Base': 'Information repository that provides a means for information to be collected, organized, shared, searched and utilized.  Allows customers to get answers to common questions.',
    'Customer Support:Call Center Features': 'Allows customer support professionals access to all information required to support the customer including customer information, case history and related customer social activity.  Common features include:  call recording, analytics, workforce management, call script management, and compliance management.',
    'Customer Support:Support Analytics': 'Enables analysis of customer support activities to optimize customer support professionals, processes and tools.',
    'Reporting & Analytics:Reporting': 'Enables reporting of all data contained within the system.  Typically contains standard reports as well as the ability to create ad-hoc reports.',
    'Reporting & Analytics:Dashboards': 'An easy to read, often single page, real-time user interface, showing a graphical presentation of the current status and historical trends of an organization’s Key Performance Indicators (KPIs) to enable instantaneous and informed decisions to be made at a glance',
    'Reporting & Analytics:Forecasting': "Enables projection of sales revenue, based on historical sales data, analysis of market surveys and trends, and salespersons' estimates.",
    'Mobile & Social:Social Collaboration Features': "Enables multiple users to interact by sharing information to achieve a common goal.  Social collaboration focus's on the identification of groups and collaboration spaces in which messages are explicitly directed at the group and the group activity feed is seen the same way by everyone",
    'Mobile & Social:Social Network Integration': 'Use public social networks to listen and engage with customers.  Allows users to filter what’s important and respond quickly.  Often this functionality allows questions and requests from customers on social networks to case management.',
    'Mobile & Social:Mobile User Support': 'Allows software to be easily used on multiple mobile devices include phone and tablet devices.',
    'Platform:Customization': 'Allows administrators to customize to accomodate their unique processes.  Includes ability to create custom objects, fields, rules, calculations, and views.',
    'Platform:Workflow Capability': 'Automates a process that requires a series of steps that typically require intervention by a several different users.  Administrators can write rules to determine who and when a user needs to complete a step.  Also includes notification of users when they need to take action.',
    'Platform:User, Role, and Access Management': 'Grant access to select data, features, objects, etc. based on the users, user role, groups, etc.',
    'Platform:Internationalization': 'Enables users to view and transact business with the same content in multiple languages and currencies.',
    'Platform:Sandbox / Test Environments': 'Allows administrators to easily develop and test changes to the CRM deployment.  After changes are made admins can easily migrate the changes into the "live" or "production" environment.',
    'Platform:Document & Content Mgmt.': 'Allows consuming, publishing and editing content from a central interface.  Content management for CRM systems might include presentations, documents, images and other related electronic files.',
    'Platform:Performance and Reliability': 'Software is consistently available (uptime) and allows users to complete tasks quickly because they are not waiting for the software to respond to an action they took.',
    'Platform:Output Document Generation': 'Allows adminstrators to create templates that enable users to quickly generate dynamic documents in various formats based on the data stored in the application.',
    'Integration:Data Import & Export Tools': 'Ability to input, modify and extract data from the application in bulk through a structured file.',
    'Integration:Integration APIs': "Application Programming Interface - Specification for how the application communicates with other software.  API's typically enable integration of data, logic, objects, etc with other software applications.",
    'Integration:Breadth of Partner Applications': 'To what extent are there partner applications readily available for integrating into this product?  Partner applications typically provide complementary, best of breed functionality not offered natively in this product.'}

# for rev in df['body']:
#     for key, val in feat_dict.items():
#         print(val)

from collections import Counter

punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'
df['new_body'] = df['body'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

# convert text to lowercase
df['new_body'] = df['new_body'].str.lower()

stop_words = set(stopwords.words('english'))
char = ['.', ',', '"', "'", '/', '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '&', '-', 'management', 'mgmt',
        'user', 'support']
stop_words.update(char)  # remove it if you need punctuation
list_of_words = list()
for doc in df['new_body']:
    token = wordpunct_tokenize(doc)
    lam_token = [lemmatizer.lemmatize(i) for i in token]
    list_of_words.append([i for i in lam_token if i not in stop_words])
df['body_tokens'] = list_of_words

# most frequent words
list_of_words = list(itertools.chain(*list_of_words))
s = " "
data_set = s.join(list_of_words)

# split() returns list of all the words in the string
split_it = data_set.split()
# split_it = [lemmatizer.lemmatize(word) for word in split_it]
# Pass the split_it list to instance of Counter class.
Counter = Counter(split_it)

# most_common() produces k frequently encountered
# input values and their respective counts.
most_occur = dict(Counter.most_common(100))
most_occur_100 = [k for k in most_occur.keys()]
keyword = keywords(s.join(most_occur_100)).split('\n')
stop_100 = list(set(most_occur_100) - set(keyword))
print(stop_100)

feat_dict_updated = dict()
for key, val in feat_dict.items():
    cat = [lemmatizer.lemmatize(i.lower()) for i in wordpunct_tokenize(key.split(":")[1]) if
           i.lower() not in stop_words]
    w = [i.lower() for i in wordpunct_tokenize(val) if i.lower() not in stop_words]
    w = list(set([lemmatizer.lemmatize(word) for word in w]))
    w = [i for i in w if i not in stop_100]
    feat_dict_updated[key] = (cat, w)

for index, row in df.iterrows():
    r = row['body_tokens']
    for key, val in feat_dict_updated.items():
        add_score = 0
        first_key = val[0]
        second_key = val[1]
        for item in first_key:
            if r.count(item) != 0:
                add_score += 0.20
        for item in second_key:
            if r.count(item) != 0:
                add_score += 0.10

        if df.at[index, key] + add_score > 0.96:
            df.at[index, key] = round(random.uniform(0.95, 0.96) * 100, 2)
        else:
            df.at[index, key] = round((df.at[index, key] + add_score) * 100, 2)

# df = df.drop(['body_tokens', 'new_body'], axis=1)
# df.to_csv("results/tuned_elmo.csv", index=False)
######################################################################################################################
