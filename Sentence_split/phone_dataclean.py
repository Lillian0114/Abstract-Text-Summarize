import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import csv
import string
# import matplotlib.pyplot as plt
from tqdm import tqdm
from textblob import TextBlob
import string
import re
import html
from nltk import pos_tag
from spellchecker import SpellChecker
import pickle   
import nltk


spell  = SpellChecker()
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

contractions=pickle.load(open("./dataset/contractions.pkl","rb")) # isn't to is not...

def sentiment_analysis(reviewText, subject_threshold):
    blob = TextBlob(reviewText)
    newFinalText = ""
    for i in range(len(blob.sentences)):
        subjective_value = blob.sentences[i].sentiment[1]
        if subjective_value > subject_threshold:
            newFinalText += str(blob.sentences[i]) + " "
    return newFinalText

# stopwords
def load_stopwords(path):
    stop_words = []
    f = open(path, "r")
    for line in f:
        line = line.strip('\n').strip()
        stop_words.append(line)
    return stop_words

def cleaning_data2(review_sentence, productName, stopwords):
    productName = productName.translate(str.maketrans('', '', string.punctuation)).lower().split()

    review_clean = ""
    review_sentence = re.sub(r'https?:\/\/.*[\r\n]*', '', review_sentence, flags=re.MULTILINE)
    tmp = TextBlob(review_sentence).tags
    # words = review_sentence.split()
    # tmp = pos_tag(words)
    
    for (word, pos) in tmp:
        word = word.lower()
        # html entity decode
        word = html.unescape(word)
        
        #remove product name
        if word in productName:
            word = " "
            continue
        elif word in contractions: # isn't to is not...
            word = " "
            continue
        
        # remove the punctuation
        word = word.translate(str.maketrans('', '', string.punctuation))
        # remove tokens that don't contain letters or numbers  
        word = re.sub(r"[^A-Za-z0-9`]", " ", word)
        if word == "":
            word = " "
            continue

        # remove stopwords 
        if word in stopwords:
            word = " "
            continue
        
        if spell.correction(word) is not None:
            word = spell.correction(word)
        
        if not str.isalpha(word):
            word = " "
            continue

        # lemmatizer
        if pos in ['VBG','VBD','VBN','VBP','VBZ']:
            word = lemmatizer.lemmatize(word,'v')
        
        if pos in ["NNS","NNPS"]:
            word = lemmatizer.lemmatize(word,'n')
        
        if pos in ['JJ','JJR','JJS']:
            word = lemmatizer.lemmatize(word,'a')
        
        review_clean += word + " "
    return review_clean


def main():
    cwd = os.getcwd() #返回當前目錄
    filename = "20191226reviews.csv"
    filename2 = "20191226items.csv"
    stopwordname = "stopwords.txt"
    txtPath = os.path.join(cwd,'dataset','cellphone',filename) 
    txtPath2 = os.path.join(cwd,'dataset','cellphone',filename2) 
    txtPath3 = os.path.join(cwd,'dataset',f'{stopwordname}') 
    
    reviews = pd.read_csv(txtPath) #67986
    print("origin review:" ,reviews.shape)
    items = pd.read_csv(txtPath2)
    items.drop(columns=['url', 'image','rating','reviewUrl','totalReviews','originalPrice'], axis=1, inplace=True)
    items.rename(columns={"title": "productName"}, inplace=True)
    # reviews = reviews[reviews['helpfulVotes'].notna()]
    reviews = reviews[reviews['title'].notna()]
    reviews = reviews[reviews['body'].notna()]
    items = items[items['productName'].notna()]
    items = items[items['brand'].notna()]
    review_data = pd.merge(reviews, items, on ='asin')
    print("Merge review:" ,review_data.shape) # 67756

    stop_words = load_stopwords(txtPath3)
    custom_stop = ['another','anybody',"anyhow",'anyone','anything','anyway','anyways','anywhere','phone']
    stop_words = stop_words
    # stop_words.remove('not')
    stop_words.extend(custom_stop) 
    keywords = review_data["brand"].apply(lambda x: x.lower()).unique().tolist()
    stop_words.extend(keywords) 

    subject_threshold = 0.05
    review_data['review_clean'] = reviews['body']
    remove_index = []
    for i in tqdm(range(len(review_data))):
        per_reviewText = review_data['body'][i]
        # print(per_reviewText)
        per_reviewText = str(per_reviewText)
        per_review_subjective = sentiment_analysis(per_reviewText, subject_threshold)
        if per_review_subjective != "":
            review_data['review_clean'][i] = per_review_subjective
            # if not pd.isnull(review_data['brand'][i]) :
            productName = review_data['productName'][i].lower()
            # else:
                # product_brand = ""
            review_data['review_clean'][i] = cleaning_data2(per_review_subjective, productName, stop_words)
        else:
            # review_data.drop([i], axis=0, inplace = True)
            # review_data['review_clean'][i] = ""
            remove_index.append(i)
    
    review_data.drop(remove_index, axis=0, inplace = True)
    print("Merge review:" ,review_data.shape) # 67756


    for i in range(101,150):
        print("Review #",i+1)
        print("Review:",review_data['review_clean'].iloc[[i]])
        print("Original Review:",review_data['body'].iloc[[i]])
        print("\n")

     
    groupby_csvpath = os.path.join(cwd,'dataset','cellphone','phone0323_dataclean.csv') 
    review_data.sort_values(by=['asin'])
    review_data.to_csv(groupby_csvpath, index=False)

if __name__ == '__main__':
    main()