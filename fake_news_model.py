# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:01:47 2021

@author: agent
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import nltk

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

# Add flag to track fake and real
fake['target'] = 'fake'
true['target'] = 'true'

# Concatenate dataframes
data = pd.concat([fake, true]).reset_index(drop = True)
data.shape

# Shuffle the data
from sklearn.utils import shuffle
data = shuffle(data)
data = data.reset_index(drop=True)

# Removing the date (we won't use it for the analysis)
data.drop(["date"],axis=1,inplace=True)

# Removing the title (we will only use the text)
data.drop(["title"],axis=1,inplace=True)

# Convert to lowercase
data['text'] = data['text'].apply(lambda x: x.lower())

# Remove punctuation
import string

def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str

data['text'] = data['text'].apply(punctuation_removal)

# Removing stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')

data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    
# # Split the data
X_train,X_test,y_train,y_test = train_test_split(data['text'], data.target, test_size=0.2, random_state=42)

# from sklearn.tree import DecisionTreeClassifier

# # Vectorizing and applying TF-IDF
# pipe = Pipeline([('vect', CountVectorizer()),
#                  ('tfidf', TfidfTransformer()),
#                  ('model', DecisionTreeClassifier(criterion= 'entropy',
#                                            max_depth = 20, 
#                                            splitter='best', 
#                                            random_state=42))])
# # Fitting the model
# model = pipe.fit(X_train, y_train)

# # Accuracy
# prediction = model.predict(X_test)


# print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
# print(prediction)

# Vectorizing and applying TF-IDF
from sklearn.linear_model import LogisticRegression
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', LogisticRegression())])
# Fitting the model
model = pipe.fit(X_train, y_train)
# Accuracy
#prediction = model.predict(X_test)
#print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

    