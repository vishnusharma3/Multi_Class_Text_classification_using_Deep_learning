# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:25:08 2020

@author: ashis
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from text_cleaning import *
cl=text_cleaning()
import pandas as pd
import sys,os
from pathlib import Path, PurePath
sys.path.append(PurePath(Path(__file__).parents[1]).as_posix())
from sklearn import preprocessing
import pickle
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from keras import models
from keras import layers
#from keras.callbacks import EarlyStopping, ModelCheckpoint

# Importing the dataset
input_data =pd.read_excel(str(Path(str(PurePath(Path(__file__).parents[1])) + os.sep +"Input-data" + os.sep))+os.sep+'data.xlsx',dtype={'class':str})

y = input_data['class']
X =input_data['text']


input_data.loc[:,'text']= [cl.data_cleaning(str(x)) for x in input_data.loc[:,'text']]
input_data=input_data[input_data['text']!='']
cleaned_data=input_data.dropna()
cleaned_data['class'].value_counts()

cleaned_data.groupby(["class"])
df3=cleaned_data.reset_index(drop=True)

df4=df3.groupby('class').head(100)
df4=df4.reset_index(drop=True)
df4['class'].value_counts()
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df4['class']=label_encoder.fit_transform(df4['class'])
#pickle.dump(self.label_encoder,open(deeplearning_training.PKL_PATH+r'label_encoder.pkl','wb'))
cleaned_data.head()
df4=df4[['class','text']]
df4.to_excel('cleaned_data.xlsx')


x_train, x_val,y_train, y_val = train_test_split(df4['text'], df4['class'], test_size=0.25)
x_train.head()
tfidf_vectorizer= TfidfVectorizer(min_df=1,ngram_range=(1, 1))
train_input = tfidf_vectorizer.fit_transform(x_train)
train_output=y_train
#train_output = to_categorical(y_train)
#pickle.dump(tfidf_vectorizer,open(deeplearning_training.PKL_PATH+r'tfidf_vectorizer.pkl','wb'))
validation_input = tfidf_vectorizer.transform(x_val)
#validation_output = to_categorical(y_val)
validation_output = y_val

print(train_output)

 #####################################################################
# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import MultinomialNB,GaussianNB
classifier = MultinomialNB()
classifier.fit(train_input.toarray(), train_output)


# Predicting the Test set results
y_pred = classifier.predict(validation_input.toarray())
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(validation_output, y_pred)
print(cm)
accuracy_score(validation_output, y_pred)