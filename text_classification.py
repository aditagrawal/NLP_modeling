#!/usr/bin/python3.5
import numpy as np
import pandas as pd
import pickle
from functools import reduce #python 3
import operator

from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer 
from gensim.models import Word2Vec

import tensorflow as tf
from tensorflow import keras

from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential,load_model


from keras.layers import Dense, Flatten, LSTM
from keras.layers.embeddings import Embedding

from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


# In[70]:


trainingpd=pd.read_csv("/axp/rim/mldsml/dev/adit/Learning/NLP/movie_review_sentiment_analysis/data/train.tsv",sep="\t")
trainpd, valpd = train_test_split(trainingpd, test_size=0.3)

testpd=pd.read_csv("/axp/rim/mldsml/dev/adit/Learning/NLP/movie_review_sentiment_analysis/data/test.tsv",sep="\t")
testpd["Sentiment"]=0


# In[71]:


english_words=pd.read_csv("/axp/rim/mldsml/dev/adit/Learning/NLP/datasets/englishwords.txt",names=["words"])


# In[72]:


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
def find_vocab_size(X_train,X_test):
    flatdata1 = [ item for elem in X_train for item in elem]
    flatdata2 = [ item for elem in X_test for item in elem]
    flatdata1.extend(flatdata2)
    list_set = list(set(flatdata1))
    vocab_size=len(list_set)
    return vocab_size

def load_object(filname):
    with open(filname, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

def tokenize_data(datum):
    toks=[row.replace(",", " , ").replace(".", " . ").replace("  ", " ").split(' ') for row in datum]
    processed_list=[]
    for row in toks:
        rand=[s.lower() for s in row]
        processed_list.append(rand)
    return processed_list

def pre_process(toks):
    ps = PorterStemmer()
    processed_list=[]
    for row in toks:
        rand=[s.lower() for s in row]
        rand=[ps.stem(s) for s in rand]
        processed_list.append(rand)
    return processed_list

def remove_stopwords(tok):
    engstopwords=pd.read_csv("/axp/rim/mldsml/dev/adit/Learning/NLP/datasets/stopwords/english",names=["stops"])
    engstopwords_set=set(np.array(engstopwords.stops))
    rand2=[]
    for row in tok:
        rand1=[]
        for word in row:
            if word not in engstopwords_set:
                rand1.append(word)
        rand2.append(rand1)
    return rand2

def remove_most_least_freq_words(tok,word_index_map,top=3,bottom=3):
    rand2=[]
    for row in tok:
        rand1=[]
        for word in row:
            freq=word_index_map[word]
            if freq>top and freq<(len(word_index_map)-bottom):
                rand1.append(word)
        rand2.append(rand1)   
    return rand2

def remove_words_basis_freq(tok,word_freq_map,min_freq=2,max_freq=6):
    rand2=[]
    for row in tok:
        rand1=[]
        for word in row:
            freq=word_freq_map[word]
            if freq>=min_freq and freq<=max_freq:
                rand1.append(word)
        rand2.append(rand1)   
    return rand2
        
def to_categ(y_lables):
    return to_categorical(y_lables)

def word_index_freq(data1,data2):
    flatdata1 = [ item for elem in data1 for item in elem]
    print("list of list 1 to list done")
    flatdata2 = [ item for elem in data2 for item in elem]
    print("list of list 2 to list done")
    flatdata1.extend(flatdata2)
    print("list appended")
    list_set = list(set(flatdata1))
    print("listed")
    word_freq_map = {x:flatdata1.count(x) for x in list_set}
    print("frequency done")
    #word_freq_map = {} 
    #for items in my_list: 
    #    word_freq_map[items] = my_list.count(items)
    word_index_map={key: rank for rank, key in enumerate(sorted(word_freq_map, key=word_freq_map.get, reverse=True), 1)}
    print("ranking done")
    return word_index_map, word_freq_map

def simple_word_index_freq(data1,data2):
    flatdata1 = [ item for elem in data1 for item in elem]
    flatdata2 = [ item for elem in data2 for item in elem]
    flatdata1.extend(flatdata2)
    list_set = list(set(flatdata1))
    for i in range(len(list_set)):
        word_index_map = {x:(i+1) for x in list_set}
    return word_index_map

def spell_corrector(toks,vocab):
    vocab_lower=[]
    for word in vocab:
        vocab_lower.extend(word.lower())
    vocab_lower=set(vocab_lower)
    for row in toks:
        for word in row:
            word=word.lower()
            if word in vocab_lower:
                pass
            else:
                pass
                #word assigned to closest word found in vocablower based on distance
    return toks

def assign_index_to_words(toks,word_index_map):
    rand2=[]
    for row in toks:
        rand1=[]
        for word in row:
            rand1.append(word_index_map[word])
        rand2.append(rand1)
    return rand2


def gen_submission_csv(df,scores,location):
    randdf=pd.DataFrame()
    randdf["PhraseId"]=df.PhraseId.values
    randdf["Sentiment"]=scores
    randdf.to_csv(location,index=False)


# In[73]:


def cnn_model_train(X_train,Y_train1,model_location,maxlen=10,vocab_size=19422):

    maxlen=maxlen
    vocab_size=vocab_size
    Y_train=to_categ(Y_train1)
    
    X_train = pad_sequences(X_train, maxlen=maxlen,padding="pre",truncating="pre")
    #X_test = pad_sequences(X_test, maxlen=maxlen,padding="pre",truncating="pre")

    model = Sequential()
    model.add(Embedding(vocab_size, 128, input_length=maxlen,mask_zero=True))
    model.add(Dropout(0.2))
    model.add(Conv1D(32, 3, padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D())
    model.add(Conv1D(64, 3, padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(5, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    #model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=2, batch_size=64, verbose=2)
    model.fit(X_train, Y_train, epochs=20, batch_size=64, verbose=1)
    model.save(model_location)
    # Final evaluation of the model
    #scores = model.evaluate(X_test, Y_test, verbose=0)
    #print("Accuracy: %.2f%%" % (scores[1]*100))
    return model


# In[74]:


def rnn_model_train(X_train,Y_train1,model_location,maxlen=10,vocab_size=19422):

    maxlen=maxlen
    vocab_size=vocab_size
    
    X_train = pad_sequences(X_train, maxlen=maxlen,padding="pre",truncating="pre")
    #X_test = pad_sequences(X_test, maxlen=maxlen,padding="pre",truncating="pre")
    Y_train=to_categ(Y_train1)

    model = Sequential()
    model.add(Embedding(vocab_size, 128, input_length=maxlen,mask_zero=True))
    model.add(LSTM(64,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))
    model.add(LSTM(32,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))
    model.add(Dense(5, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    #model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=2, batch_size=64, verbose=2)
    model.fit(X_train, Y_train, epochs=2, batch_size=64, verbose=1)
    model.save(model_location)
    # Final evaluation of the model
    #scores = model.evaluate(X_test, Y_test, verbose=0)
    #print("Accuracy: %.2f%%" % (scores[1]*100))
    return model


# In[75]:


def rnn_model_score(X_test,model_location,scores_location,maxlen=10):
    maxlen=maxlen

    X_test = pad_sequences(X_test, maxlen=maxlen,padding="pre",truncating="pre")
    model=load_model(model_location)
    scores=model.predict_classes(X_test)
    save_object(scores, scores_location)
    return scores


# In[76]:


def cnn_model_score(X_test,model_location,scores_location,maxlen=10):
    maxlen=maxlen

    X_test = pad_sequences(X_test, maxlen=maxlen,padding="pre",truncating="pre")
    model=load_model(model_location)
    scores=model.predict_classes(X_test)
    save_object(scores, scores_location)
    return scores


# In[77]:


def train_model_w2v(X_train,model_location,sg=1,embed_size=128):
    rand=tokenize_data(X_train)
    model = Word2Vec(rand, min_count=1,size=embed_size)
    model.save(model_location)
    return model


# In[78]:


def score_model_w2v(X_test,model_location,embed_location):
    rand2=[]
    rand=tokenize_data(X_test)
    model = Word2Vec.load(model_location)
    for row in rand:
        rand1=[]
        for word in row:
            word_embed=model[word]
            rand1.append(word_embed)
        rand1_mean=list(np.mean(rand1, axis=0))
        rand2.append(rand1_mean)
    rand2=np.array(rand2)
    save_object(rand2, embed_location)
    return rand2


# In[79]:


def dense_model_train(X_train,Y_train,model_location):
    Y_train=to_categ(Y_train)
    model = Sequential()
    model.add(Dense(64, activation='relu',input_shape=(np.shape(X_train)[1],)))
    model.add(Dense(5, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    model.fit(X_train, Y_train, epochs=2, batch_size=64, verbose=2)
    model.save(model_location)
    return model


# In[80]:


def dense_model_score(X_test,model_location,scores_location):
    model=load_model(model_location)
    scores=model.predict_classes(X_test)
    save_object(scores, scores_location)
    return scores


# In[81]:


X_train=tokenize_data(trainingpd.Phrase.values)
X_test=tokenize_data(testpd.Phrase.values)


# In[82]:


#flatdata1 = [ item for elem in X_train for item in elem]
#flatdata2 = [ item for elem in X_test for item in elem]
#flatdata1.extend(flatdata2)
#list_set = list(set(flatdata1))
#len(list_set)


# In[50]:


Y_train=np.array(trainingpd.Sentiment.values)


# In[53]:


#word_index_map, word_freq_map=word_index_freq(X_train,X_test)
word_index_map=simple_word_index_freq(X_train,X_test)

#Y_test=to_categ(testpd.Sentiment.values)

#X_train=remove_words_basis_freq(X_train,word_freq_map,min_freq=2,max_freq=6)
#X_test=remove_words_basis_freq(X_test,word_freq_map,min_freq=2,max_freq=6)

#X_train=remove_most_least_freq_words(X_train,word_index_map,top=2,bottom=3)
#X_test=remove_most_least_freq_words(X_test,word_index_map,top=2,bottom=3)


# In[84]:


np.shape(trainingpd)


# In[54]:


save_object(word_index_map,"/axp/rim/mldsml/dev/adit/Learning/NLP/movie_review_sentiment_analysis/v1/word_index_map.pkl")

save_object(word_freq_map,"/axp/rim/mldsml/dev/adit/Learning/NLP/movie_review_sentiment_analysis/v1/word_freq_map.pkl")


# In[ ]:


X_train=assign_index_to_words(X_train,word_index_map)
X_test=assign_index_to_words(X_test,word_index_map)


# In[ ]:


X_train=np.array(X_train)
X_test=np.array(X_test)


# In[ ]:


save_object(X_train,"/axp/rim/mldsml/dev/adit/Learning/NLP/movie_review_sentiment_analysis/v1/X_train")
save_object(Y_train,"/axp/rim/mldsml/dev/adit/Learning/NLP/movie_review_sentiment_analysis/v1/X_train")
save_object(X_test,"/axp/rim/mldsml/dev/adit/Learning/NLP/movie_review_sentiment_analysis/v1/X_train")


# In[ ]:


rnn_model=rnn_model_train(X_train,Y_train,model_location="/axp/rim/mldsml/dev/adit/Learning/NLP/movie_review_sentiment_analysis/v1/rnn_nlp.model")
rnn_scores=rnn_model_score(X_test,model_location="/axp/rim/mldsml/dev/adit/Learning/NLP/movie_review_sentiment_analysis/v1/rnn_nlp.model",scores_location="/axp/rim/mldsml/dev/adit/Learning/NLP/movie_review_sentiment_analysis/v1/rnn_nlp_scores.pkl")


# In[ ]:


cnn_model=cnn_model_train(X_train,Y_train,model_location="/axp/rim/mldsml/dev/adit/Learning/NLP/movie_review_sentiment_analysis/v1/cnn_nlp.model")
cnn_scores=cnn_model_score(X_test,model_location="/axp/rim/mldsml/dev/adit/Learning/NLP/movie_review_sentiment_analysis/v1/cnn_nlp.model",scores_location="/axp/rim/mldsml/dev/adit/Learning/NLP/movie_review_sentiment_analysis/v1/cnn_nlp_scores.pkl")


# In[ ]:


gen_submission_csv(testpd,rnn_scores,location="/axp/rim/mldsml/dev/adit/Learning/NLP/movie_review_sentiment_analysis/v1/rnn_submission.csv")

gen_submission_csv(testpd,cnn_scores,location="/axp/rim/mldsml/dev/adit/Learning/NLP/movie_review_sentiment_analysis/v1/cnn_submission.csv")


# In[ ]:


w2vcorpus=list(trainingpd.Phrase.values)
rn2=list(testpd.Phrase.values)
w2vcorpus.extend(rn2)
w2vcorpus=np.array(w2vcorpus)


train_model_w2v(w2vcorpus,model_location="/axp/rim/mldsml/dev/adit/Learning/NLP/movie_review_sentiment_analysis/v1/w2v_nlp.model",sg=1,embed_size=128)


# In[ ]:


score_file_train=score_model_w2v(trainingpd.Phrase.values,model_location="/axp/rim/mldsml/dev/adit/Learning/NLP/movie_review_sentiment_analysis/v1/w2v_nlp.model",embed_location="/axp/rim/mldsml/dev/adit/Learning/NLP/movie_review_sentiment_analysis/v1/w2v_nlp_train_embeds.pkl")

score_file_test=score_model_w2v(testpd.Phrase.values,model_location="/axp/rim/mldsml/dev/adit/Learning/NLP/movie_review_sentiment_analysis/v1/w2v_nlp.model",embed_location="/axp/rim/mldsml/dev/adit/Learning/NLP/movie_review_sentiment_analysis/v1/w2v_nlp_test_embeds.pkl")

Y_train=np.array(trainingpd.Sentiment.values)

dense_model=dense_model_train(score_file_train,Y_train,model_location="/axp/rim/mldsml/dev/adit/Learning/NLP/movie_review_sentiment_analysis/v1/dense.model")
dense_scores=dense_model_score(score_file_test,model_location="/axp/rim/mldsml/dev/adit/Learning/NLP/movie_review_sentiment_analysis/v1/dense.model",scores_location="/axp/rim/mldsml/dev/adit/Learning/NLP/movie_review_sentiment_analysis/v1/dense_model_scores.pkl")


# In[319]:


gen_submission_csv(testpd,dense_scores,location="/axp/rim/mldsml/dev/adit/Learning/NLP/movie_review_sentiment_analysis/v1/w2v_submission.csv")



