# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 02:54:03 2016

@author: Ronak
"""
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


train=pd.read_csv('C:\Users\Ronak\Desktop\Important\labeledTrainData.tsv', header=0,delimiter="\t",quoting=3)
def CleanReview(raw_review):
    #Removing HTML tags
    review_text= BeautifulSoup(raw_review).get_text()
    #Picking up only letters
    letters= re.sub("[^a-zA-Z]"," ",review_text)
    #Splitting up into words of lower case only
    words=letters.lower().split()
    #Converting stopwords into sets as sets are faster to search than lists
    sets= set(stopwords.words("english"))
    #Removing stop words
    words=[w for w in words if not w in sets]
    #Joining it all together
    return(" ".join(words))
length= train["review"].size
clean_train=[]
for i in xrange(0,length):
    clean_train.append(CleanReview(train["review"][i]))
vectorizer=CountVectorizer(analyzer="word", \
                           tokenizer=None, \
                           preprocessor=None, \
                           stop_words=None, \
                           max_features=5000)
train_data_features=vectorizer.fit_transform(clean_train)
train_data_features= train_data_features.toarray()
vocab= vectorizer.get_feature_names()


forest=RandomForestClassifier(n_estimators=100)
forest= forest.fit(train_data_features,train["sentiment"])
test= pd.read_csv(r'C:\Users\Ronak\Desktop\Important\testData.tsv',header=0,delimiter="\t", quoting=3)
num_rev= len(test["review"])
clean_test=[]
for i in xrange(0,num_rev):
    clean_test.append(CleanReview(test["review"][i]))
test_data_features=vectorizer.transform(clean_test)
test_data_features=test_data_features.toarray()

result= forest.predict(test_data_features)
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )

    
    
    