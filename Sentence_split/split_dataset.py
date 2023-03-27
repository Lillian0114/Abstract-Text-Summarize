import nltk
from nltk.probability import FreqDist
import os
import pandas as pd
import numpy as np
from tqdm import tqdm, tqdm_pandas
import string
import numpy as np, codecs, json, pickle, sys
import matplotlib.pyplot as plt
# from apyori import apriori
from collections import defaultdict
from scipy.sparse import dok_matrix
import gzip
import re
import html
from spellchecker import SpellChecker
tqdm.pandas()
from sklearn.model_selection import train_test_split

# spell  = SpellChecker()
# lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

# load dataset
def load_Preprocess_review(file_path):
    with open(file_path, 'r', encoding="utf-8",errors='ignore') as file:
        group_review = pd.read_csv(file)
    return group_review

def split_dataset(df):
    pos_df = df[df['rating'] > 3]
    neg_df = df[df['rating'] < 3]
    return pos_df,neg_df

def main():
    cwd = os.getcwd() 
    filename = "0325phone_segment.csv"
    file1 = "phone_train_pos.csv"
    file2 = "phone_test_pos.csv"
    file4 = "phone_train_neg.csv"
    file3 = "phone_test_neg.csv"
    txtPath = os.path.join(cwd,filename) 
    txtPath2 = os.path.join(cwd,'dataset','training_testing') 
    review_data = load_Preprocess_review(txtPath) 
    review_data['rating'] = review_data["rating"].astype(int)
    print(len(review_data))
    pos_df, neg_df = split_dataset(review_data)
    pos_train, pos_test = train_test_split(pos_df, test_size=0.2)
    neg_train, neg_test = train_test_split(neg_df, test_size=0.2)
    pos_train.to_csv(os.path.join(file1),index=None)
    pos_test.to_csv(os.path.join(file2),index=None)
    neg_train.to_csv(os.path.join(file3),index=None)
    neg_test.to_csv(os.path.join(file4),index=None)

    print(len(pos_df),"    ",len(neg_df))
    print(len(pos_df)+len(neg_df))

if __name__ == '__main__':
    main()
