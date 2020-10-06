''' This script takes a data which contain two variables
   1- text: this is the input variable for multiclass text classification problem
   2- Class: this is the output variable  in the forms of categories'''

''' importing the necessary libraries and scripts'''
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
from keras.callbacks import EarlyStopping, ModelCheckpoint
sys.path.insert(1,str(Path(str(PurePath(Path(__file__).parents[1])) + os.sep +"Cleaning" + os.sep)) + os.sep)
from text_cleaning import *
cl=text_cleaning()


uncleaned_data=pd.read_csv(str(Path(str(PurePath(Path(__file__).parents[1])) + os.sep +"Input-data" + os.sep))+os.sep+'data.csv',dtype={'class':str})

class deeplearning_training():
    ''' class to produce the model files after training'''

    def __init__(self,uncleaned_data):
        self.input_data=uncleaned_data
        return None

    @classmethod
    def set_pkl_path(cls):
        '''This classmethod sets PKL PATH from
        which all the pkl,json,h5 files will be dumped'''

        cls.PKL_PATH = str(Path(str(PurePath(Path(__file__).parents[1])) + os.sep +"Model_files" + os.sep)) + os.sep

    def cleaning(self):
        ''' Apply cleaning steps on the input of uncleaned data'''

        self.input_data.loc[:,'text']= [cl.data_cleaning(str(x)) for x in self.input_data.loc[:,'text']]
        self.input_data=self.input_data[self.input_data['text']!='']
        self.cleaned_data=self.input_data.dropna()
        return self.cleaned_data

    def label_enconding_ontarget(self):

        ''' This method applies label enconding on the target variable'''

        self.cleaned_data=self.cleaning()
        self.label_encoder = preprocessing.LabelEncoder()
        self.cleaned_data['class']=self.label_encoder.fit_transform(self.cleaned_data['class'])
        pickle.dump(self.label_encoder,open(deeplearning_training.PKL_PATH+r'label_encoder.pkl','wb'))
        return self.cleaned_data
    
    def train_test_splitter(self):

        '''This method splits the cleaned data into training and test data. The propostion is given
        as a paramter which decides the training and test sets size'''
        
        self.cleaned_data=self.label_enconding_ontarget()
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.cleaned_data['text'], self.cleaned_data['class'], test_size=0.25)
        return self.x_train, self.x_val, self.y_train, self.y_val

    def vectorizer_tfidf(self):

        ''' Creating tfidf_vectorizer for training and validation set'''

        self.x_train, self.x_val, self.y_train, self.y_val=self.train_test_splitter()
        tfidf_vectorizer= TfidfVectorizer(min_df=1,ngram_range=(1, 1))
        self.train_input = tfidf_vectorizer.fit_transform(self.x_train)
        self.train_output = to_categorical(self.y_train)
        pickle.dump(tfidf_vectorizer,open(deeplearning_training.PKL_PATH+r'tfidf_vectorizer.pkl','wb'))
        self.validation_input = tfidf_vectorizer.transform(self.x_val)
        self.validation_output = to_categorical(self.y_val)
        return self.train_input,self.train_output,self.validation_input,self.validation_output

    def model_fully_connected_layers(self):

        '''Settting up fully connected model structure with valid input, hidden, output layers'''

        self.train_input,self.train_output,self.validation_input,self.validation_output=self.vectorizer_tfidf()
        self.model = models.Sequential()
        x=self.train_input.shape[1]
        y=self.train_output.shape[1]
        self.model.add(layers.Dense(int(x/3), activation='relu', input_shape=(x,)))
        self.model.add(layers.Dense(int(x/6), activation='relu'))
        self.model.add(layers.Dense(y, activation='softmax'))
        self.model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        return self.model

    def callbacks_parameter(self):

       ''' Eearly stopping parameter decides at what phase the training should be stopped to avoid overfitting'''

       self.callbacks = [EarlyStopping(monitor='val_acc', patience=3)]
       return self.callbacks
    
    def training(self):

        '''Fitting training and validatation set with fully connected model i.e training the deep learning model'''

        self.model_fully_connected=self.model_fully_connected_layers()
        self.model_fully_connected.fit(self.train_input,self.train_output,epochs=40,batch_size=512,callbacks=self.callbacks_parameter(),validation_data=(self.validation_input,self.validation_output))
        with open(deeplearning_training.PKL_PATH+"model_fully_connected.json", "w") as json_file:
            json_file.write(self.model_fully_connected.to_json())
        self.model_fully_connected.save_weights(deeplearning_training.PKL_PATH+"model_fully_connected.h5")
        return self.model_fully_connected


deeplearning_training.set_pkl_path()
deeplearning_training(uncleaned_data).training()