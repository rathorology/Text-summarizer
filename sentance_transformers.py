"""
This basic example loads a pre-trained model from the web and uses it to
generate sentence embeddings for a given list of sentences.
"""

from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import logging

#### Just some code to print debug information to stdout
np.set_printoptions(threshold=100)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


# Load Sentence model (based on BERT) from URL
model = SentenceTransformer('bert-base-nli-mean-tokens')

# sentences = ['This framework generates embeddings for each input sentence',
#     'Sentences are passed as a list of string.',
#     'The quick brown fox jumps over the lazy dog.']
# sentence_embeddings = model.encode(sentences)

s = [
    "Great Solution For Team Collaboration & Basic Project Management.I've used Basecamp to manage development projects, marketing programs, product management, company calendars, and basic HR/company policy documentation for several years. I've always found it to be reliable and easy to use. I will continue to use it going forward..Basecamp is incredibly intuitive and easy-to-use. New users can start using the system in a very short period of time. I love the ability to upload files to specific projects or teams and to organize them into folders and update them with new versions. I also really appreciate how Basecamp makes sharing news, messages, and schedules with members of the team. Basecamp is also very affordable, even for small teams..I would like to see some additional project management features added. Specifically, project milestones and Gantt charts. A few of the features are not particularly useful, such as the Gaussian-looking completion chart."]

doc = model.encode(s)[0]
doc1 = model.encode(['Compliance Management'])[0]
doc2 = model.encode(['Credit Risk Management'])[0]
doc3 = model.encode(['For Hedge Funds'])[0]
doc4 = model.encode(['Liquidity Analysis'])[0]
doc5 = model.encode(['Loan Portfolio Management'])[0]
doc6 = model.encode(['Market Risk Management'])[0]
doc7 = model.encode(['Operational Risk Management'])[0]
doc8 = model.encode(['Portfolio Management  '])[0]
doc9 = model.encode(['Portfolio Modeling'])[0]
doc10 = model.encode(['Risk Analytics Benchmarks'])[0]
doc11 = model.encode(['Stress Tests'])[0]
doc12 = model.encode(['Value At Risk Calculation'])[0]


def square_rooted(x):
    return np.math.sqrt(sum([a * a for a in x]))


def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return numerator / float(denominator)


print(cosine_similarity(doc, doc1))
print(cosine_similarity(doc, doc2))
print(cosine_similarity(doc, doc3))
print(cosine_similarity(doc, doc4))
print(cosine_similarity(doc, doc5))
print(cosine_similarity(doc, doc6))
print(cosine_similarity(doc, doc7))
print(cosine_similarity(doc, doc8))
print(cosine_similarity(doc, doc9))
print(cosine_similarity(doc, doc10))
print(cosine_similarity(doc, doc11))
print(cosine_similarity(doc, doc12))

# The result is a list of sentence embeddings as numpy arrays
# for sentence, embedding in zip(sentences, sentence_embeddings):
#     print("Sentence:", sentence)
#     print("Embedding:", embedding)
#     print("")
