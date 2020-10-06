# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 08:43:46 2020

@author: ashis
"""

import pickle
from keras.models import model_from_json
from shipmentdesc_cleaning import data_cleaning_test
import pandas as pd
import numpy as np
import sys,os
from pathlib import Path, PurePath
sys.path.append(PurePath(Path(__file__).parents[1]).as_posix())
sys.path.insert(1,str(Path(str(PurePath(Path(__file__).parents[1])) + os.sep +"Cleaning" + os.sep)) + os.sep)
from text_cleaning import *
cl=text_cleaning()

model_files_path=str(Path(str(PurePath(Path(__file__).parents[1])) + os.sep +"Model_files" + os.sep)) + os.sep

json_file = open(model_files_path+'model_fully_connected.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_fully_connected = model_from_json(loaded_model_json)
model_fully_connected.load_weights(model_files_path+"model_fully_connected.h5")
label_encoder=pickle.load(open(model_files_path+r'label_encoder.pkl', 'rb'))
tfidf_vectorizer=pickle.load(open(model_files_path+r'tfidf_vectorizer.pkl', 'rb'))


def deeplearning_prediction(review):
    cleaned_review=cl.data_cleaning(review)
    vectorized_cleaned_review=tfidf_vectorizer.transform([cleaned_review]).toarray()
    class_predicted=model_fully_connected.predict(vectorized_cleaned_review)
    max_prob_class = np.argmax(class_predicted, axis=1)
    top1_class=label_encoder.inverse_transform(max_prob_class)    
    return top1_class,cleaned_review

test_data=pd.read_excel(str(Path(str(PurePath(Path(__file__).parents[1])) + os.sep +"Test-data" + os.sep))+os.sep+'test_data.xlsx',dtype={'class':str})
test_data=test_data[['text','category']]

list_of_reviews=[]
list_of_classes=[]
count=0
for review in test_data['text']:
    prediction=deeplearning_prediction(review)
    list_of_reviews.append(prediction[1])
    list_of_classes.append(prediction[0])
    count+=1
    print(count)

test_data['cleaned_reviews']=list_of_reviews
test_data['classes_predicted']=list_of_classes
test_data=test_data[test_data['cleaned_reviews']!='']
test_data['match']=np.where(test_data.classes_predicted == test_data.category, 'True', 'False')
test_data['top1accuracy']=len(test_data[test_data['match']=='True'])/len(test_data)*100



#test_data=test_data[test_data['review']!='']
#test_data=test_data.dropna()
#x_test=test_data['review']
#test_input=tfidf_vectorizer.transform(x_test)

#test_predicted = model_fully_connected.predict(test_input)
#test_predicted = np.argmax(test_predicted, axis=1)
#test_predicted=label_encoder.inverse_transform(test_predicted)

#test_data.to_excel('test_data_deep_learning_results4.xlsx')