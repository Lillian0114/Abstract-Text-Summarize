from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
import wordcloud
import pandas as pd
import os

from nltk.corpus import wordnet
from nltk import pos_tag
from apyori import apriori
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import numpy as np
import os
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from tqdm import tqdm, tqdm_pandas
import csv
from textblob import TextBlob
from nltk import tokenize
import string
import matplotlib.pyplot as plt
import math
import warnings
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
wnl = WordNetLemmatizer()
warnings.filterwarnings('ignore')
tqdm.pandas()

# load dataset
def load_Preprocess_review(file_path):
    with open(file_path, 'r', encoding="utf-8",errors='ignore') as file:
        group_review = pd.read_csv(file)

    return group_review

# stopwords
def load_stopwords(path):
    stop_words = []
    f = open(path, "r")
    for line in f:
        line = line.strip('\n').strip()
        stop_words.append(line)
    return stop_words

def noun_collector(word_tag_list):
    if(len(word_tag_list)>0):
        return [word for (word, tag) in word_tag_list if tag in {'NN', 'NNS', 'NNP', 'NNPS'}]

def n_v_a_collector(word_tag_list):
    if(len(word_tag_list)>0):
        return [word for (word, tag) in word_tag_list if tag in {'NN', 'NNS', 'NNP', 'NNPS', 'JJR', 'JJ'
                                                                 'JJS','VB', 'VBD','VBG','VBP','VBZ','VBN'}]


def main():
    cwd = os.getcwd() 
    filename = "phone0302_dataclean.csv"
    txtPath = os.path.join(cwd,'dataset','cellphone',filename) 
    review_data = load_Preprocess_review(txtPath) 
    review_data["rating"] = review_data["rating"].astype('int')
    review_data["positivity"] = review_data["rating"].apply(lambda x: 1 if x > 3 else(0 if x==3 else -1))
    # review_data['segement'] = review_data['review_clean']
    # review_data['lemm_reviewText'] = review_data['review_clean']
    review_data['review_pos'] = review_data['review_clean']
    review_data['pos_noun'] = review_data['review_clean']
    review_data['pos_noun_verb_adj'] = review_data['review_clean']
    review_data = review_data[review_data['helpfulVotes'].notna()]
    review_data = review_data[review_data['title'].notna()]
    review_data = review_data[review_data['body'].notna()]
    review_data = review_data[review_data['review_clean'].notna()]
    #=======stopwords=======
    stopwordname = "stopwords.txt"
    txtPath2 = os.path.join(cwd,'dataset',f'{stopwordname}') 
    stop_words = load_stopwords(txtPath2)
    custom_stop = ['another','anybody',"anyhow",'anyone','anything','anyway','anyways','anywhere','phone']
    stop_words = stop_words
    stop_words.remove('not')
    stop_words.extend(custom_stop) 
    keywords = review_data["brand"].apply(lambda x: x.lower()).unique().tolist()
    stop_words.extend(keywords) 
    #==============================
    
    for idx, review in tqdm(review_data.iterrows()):
        review_data.at[idx,'review_pos'] = nltk.pos_tag(review['review_pos'].split())
    
    for idx, review in tqdm(review_data.iterrows()):
        review_data.at[idx,'pos_noun'] = noun_collector(review['review_pos'])
        review_data.at[idx,'pos_noun_verb_adj'] = noun_collector(review['review_pos'])

    # print(review_data)
    filename = os.path.join(cwd,'dataset','cellphone','phone0302_pos.csv') 
    review_data.to_csv(filename,index=False, encoding="utf-8")

if __name__ == '__main__':
    main()
