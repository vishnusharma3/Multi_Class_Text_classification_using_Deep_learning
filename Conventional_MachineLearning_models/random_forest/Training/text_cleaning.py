''' importing the necessary libraries'''

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
from string import digits
eng_words = set(nltk.corpus.words.words())
import re
from nltk.corpus import wordnet as wn

class text_cleaning():
    ''' This class includes methods which are generally 
    common to any text preprocessing before vectorizing it'''
    
    def remove_digits_from_text(self,text):
        ''' Method to remove the digits from the text'''
        
        remove_digits = str.maketrans('', '', digits)
        res = text.translate(remove_digits)
        
        return res
    
    def common_text_preperation(self,text):
        ''' Method to prepare text in proper format 
        such as removing stopwords ,removing symbols etc.'''
        
        replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
        good_symbols_re = re.compile('[^0-9a-z]')
        stopwords_set = set(stopwords.words('english'))
        text = text.lower()
        text = replace_by_space_re.sub(' ', text)
        text = good_symbols_re.sub(' ', text)
        text = text.strip()
        temp = text.split()
        final_text = ' '.join([x for x in sorted(set(temp),key=temp.index) if x and x not in stopwords_set])    
        
        return(final_text) 
        
    def text_lemmatizer(self,text):
        ''' Method to Lemmatize text'''
        
        processed_text=[]
        words = word_tokenize(text)
        for w in words:
            x=lmtzr.lemmatize(w,'v')
            x=lmtzr.lemmatize(x,'n')
            processed_text.append(x)
        
        return ' '.join(processed_text)
    
    def remove_non_dictionary_words(self,text):
        ''' Method to remove the non dictionary(english) words from the text'''
        
        text = word_tokenize(text)
        l=[]
        for word in text:
             try:
                 if len(wn._morphy(word,wn.NOUN))==2:
                        l.append(wn._morphy(word,wn.NOUN)[1])
                 else:
                        l.append(wn._morphy(word,wn.NOUN)[0]) 
             except IndexError:
                        if len(wn.synsets(word))==0:
                                          None
                        else:
                            l.append(word)
        text=' '.join(l)
        return text
     
    def remove_single_character_words(self,text):
        ''' Method to remove the single character words'''
        
        processed_text = ' '.join( [w for w in text.split() if len(w)>1] )
        
        return processed_text
  
    def data_cleaning(self,text):
        '''' Method to apply all cleaning steps on the input text'''
        
        text=self.remove_digits_from_text(text)
        text=self.common_text_preperation(text)
        text=self.text_lemmatizer(text)    
        text=self.remove_non_dictionary_words(text)
        text=self.remove_single_character_words(text)
        
        return text
  