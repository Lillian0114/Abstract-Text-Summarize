import pandas as pd
import numpy as np
import os
import gzip
import pickle   
import nltk
from nltk.corpus import stopwords
import string
import re
import html
from nltk import pos_tag
from spellchecker import SpellChecker
from tqdm import tqdm
from textblob import TextBlob
from nltk import tokenize

spell  = SpellChecker()
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

contractions=pickle.load(open("./dataset/contractions.pkl","rb")) # isn't to is not...

# load dataset
def load_dataset(file_path):
    with open(file_path, 'r') as file:
        dataset = pd.read_csv(file)

    return dataset
 

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

def load_nltk_stopwords():
    # stop_words = set(stopwords.words('english'))
    stop_words = stopwords.words('english')
    return stop_words

# lemmatize 動詞變化轉為原動
def lemmatize(word):
    lemma = lemmatizer.lemmatize(word,'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word,'n')
    return lemma


def cleaning_data2(review_sentence, product_name, stopwords):
    # custom_stop = ['another','anybody',"anyhow",'anyone','anything','anyway','anyways','anywhere']
    # # stop_words = load_nltk_stopwords()
    # stop_words = stopwords
    # # stop_words.remove('not')
    # stop_words.extend(custom_stop) 
    product_name = product_name.translate(str.maketrans('', '', string.punctuation)).lower().split()

    review_clean = ""
    # sentences = tokenize.sent_tokenize(review_sentence)
    words = review_sentence.split()
    tmp = pos_tag(words)
    
    for (word, pos) in tmp:
        word = word.lower()
        # html entity decode
        word = html.unescape(word)
        
        #remove product name
        if word in product_name:
            # if any(char in string.punctuation for char in word):
            #     word = word[-1]+ " "
            # else:
            #     word = " "
            # continue
            word = " "
            continue
        
        # isn't to is not...
        if word in contractions:
            # word = contractions[word]
            word = " "
            continue
        
        # remove the punctuation
        word = word.translate(str.maketrans('', '', string.punctuation))
        # remove tokens that don't contain letters or numbers  
        word = re.sub(r"[^A-Za-z0-9`]", " ", word)
        if word =="":
            word =" "
            continue

        # remove stopwords 
        if word in stopwords:
            # if any(char in string.punctuation for char in word):
            #     word = word[-1]+ " "
            # else:
            #     word = " "
            # continue
            word = " "
            continue
        
        # if spell.correction(word) is not None and word[-1] not in string.punctuation:
        #     word = spell.correction(word)
        if spell.correction(word) is not None:
            word = spell.correction(word)

        # lemmatizer
        if pos in ['VBG','VBD','VBN']:
            word = lemmatizer.lemmatize(word,'v')
        
        if pos == "NNS":
            word = lemmatizer.lemmatize(word,'n')

        if not str.isalpha(word):
            word = " "
            continue
        
        review_clean += word + " "
    return review_clean


def main():
    # category = "Grocery_and_Gourmet_Food" # 151254
    cwd = os.getcwd() #返回當前目錄
    filename = "new0208_amazondataset.csv"
    # filename = "tmpppp.csv"
    productfilename = "product_name.csv"
    stopwordname = "stopwords.txt"
    txtPath = os.path.join(cwd,'dataset',f'{filename}') 
    txtPath1 = os.path.join(cwd,'dataset',f'{productfilename}') 
    txtPath2 = os.path.join(cwd,'dataset',f'{stopwordname}') 
    review_data = load_dataset(txtPath)
    product_name_data = load_dataset(txtPath1)
    stop_words = load_stopwords(txtPath2)
    custom_stop = ['another','anybody',"anyhow",'anyone','anything','anyway','anyways','anywhere']
    stop_words = stop_words
    stop_words.remove('not')
    stop_words.extend(custom_stop) 
    # keywords = reviews["brand"].apply(lambda x: x.lower()).unique().tolist()
    # print(len(review_data)) # 58716
    del review_data["reviewerName"]
    del review_data["unixReviewTime"]

    # for (i,per_reviewText) in tqdm(enumerate(review_data['reviewText'])):
    #     review_data['reviewText'][i] = cleaning_data2(per_reviewText)
    
    subject_threshold = 0.2
    review_data['review_clen'] = review_data['reviewText']
    for i in tqdm(range(len(review_data))):
        per_reviewText = review_data['reviewText'][i]
        per_reviewText = str(per_reviewText)
        per_review_subjective = sentiment_analysis(per_reviewText, subject_threshold)
        if per_review_subjective != "":
            per_asin = review_data['asin'][i]
            product_name = product_name_data[product_name_data["asin_num"] == per_asin]['product_name'].values[0].lower()
            review_data['review_clen'][i] = cleaning_data2(per_review_subjective, product_name, stop_words)
        else:
            review_data.drop([i], axis=0, inplace = True)
        # break
    
    groupby_csvpath = os.path.join(cwd,'dataset','food_0226_dataclean.csv') 
    review_data.sort_values(by=['asin'])
    review_data.to_csv(groupby_csvpath)

    
if __name__ == '__main__':
    main()



"""
# Loading Data 
def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def save_groupby_alldataDF():
    category = "Grocery_and_Gourmet_Food"
    cwd = os.getcwd() #返回當前目錄
    txtPath = os.path.join(cwd,'dataset',f'{category}_reviews.json.gz') 

    review_data = getDF(txtPath)
    del review_data["reviewerName"]
    del review_data["unixReviewTime"]
    review_data = review_data.astype(str)
    review_data['reviewText'] = review_data['reviewText'].apply(lambda x: '||| '+ x)
    review_data['reviewerID'] = review_data['reviewerID'].apply(lambda x: ', '+ x)
    review_data['overall'] = review_data['overall'].apply(lambda x: ', '+ x)
    review_data['summary'] = review_data['summary'].apply(lambda x: '||| '+ x)
    review_data['reviewTime'] = review_data['reviewTime'].apply(lambda x: '||| '+ x)
    # print(review_data.head(2))
    newDF_groupby_product = review_data.groupby(by='asin').sum()
    newDF_groupby_product['reviewText'] = newDF_groupby_product['reviewText'].apply(lambda x : x[4:])
    newDF_groupby_product['summary'] = newDF_groupby_product['summary'].apply(lambda x : x[4:])
    # print(newDF_groupby_product['summary'].head(3))

    groupby_csvpath = os.path.join(cwd,'dataset','groupby_asin.csv') 
    newDF_groupby_product.to_csv(groupby_csvpath)

    def cleaning_data(review_array):
    cleaning_reviews = []
    stop_words = load_nltk_stopwords()
    stop_words.remove('not')
    # stop_words.remove('but')

    for product in review_array:
        product_review = []
        for review in product:
            words = review.split()
            review_tmp = ""
            tmp = pos_tag(words)
            
            for (word, pos) in tmp:
                word = word.lower()
                # html entity decode
                word = html.unescape(word)
                
                # isn't to is not...
                if word in contractions:
                    word = contractions[word]
                
                # remove the punctuation
                word = word.translate(str.maketrans('', '', string.punctuation))
                # remove tokens that don't contain letters or numbers  
                word = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", word)
                
                # lemmatizer
                # if pos in ['VBG','VBD','VBN']:
                #     word = lemmatizer.lemmatize(word,'v')
                
                if pos == "NNS":
                    word = lemmatizer.lemmatize(word,'n')

                # remove stopwords 
                if word in stop_words:
                    word = ""
                    continue

                if not str.isalpha(word):
                    word = ""
                    continue

                if spell.correction(word) is not None:
                    word = spell.correction(word)
                
                review_tmp += word + " "

            product_review.append(review_tmp)
        cleaning_reviews.append(product_review)

    return cleaning_reviews
"""

"""
def cleaning_data(review_array):
    review_posArr = []
    review_features = []
    # cleaning_review = []

    for product in review_array:
        product_features = []
        product_pos = []
        for review in product:
            words = review.split()
            tmp = pos_tag(words)
            
            word_arr = []
            pos_arr = []
            featureTmp = {}
            for (word, pos) in tmp:
                word = word.lower()
                # html entity decode
                word = html.unescape(word)
                
                # remove the punctuation
                word = word.translate(str.maketrans('', '', string.punctuation))
                # remove tokens that don't contain letters or numbers  
                word = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", word)
                
                # lemmatizer
                word = lemmatize(word)
                
                # isn't to is not...
                if word in contractions:
                    word = contractions[word]

                # remove stopwords 
                stop_words = load_nltk_stopwords()
                if word in stop_words:
                    word = ""
                    continue

                if not str.isalpha(word):
                    word = ""
                    continue

                if spell.correction(word) is not None:
                    word = spell.correction(word)
                
                if pos.startswith('N'):
                    # featureTmp.append(word)
                    if word in featureTmp:
                        featureTmp[word] += 1
                    else:
                        featureTmp[word] = 1

                word_arr.append(word)
                pos_arr.append(pos)
            posTmp = list(zip(word_arr,pos_arr))
            # review_features.append(featureTmp)
            # review_posArr.append(posTmp)

            product_pos.append(posTmp)
            product_features.append(featureTmp)
        review_features.append(product_features)  
        review_posArr.append(product_pos)      

    return review_posArr, review_features
"""

"""

def load_PreDataSave(product_name_path, new_amazon_path, not_found_path):
    with open(product_name_path, 'r',encoding="utf-8") as file:
        products_df = pd.read_csv(file)
    with open(not_found_path, 'r',encoding="utf-8") as file:
        file_content = file.read()
        not_found_txt = file_content.split(", ")
        
    return products_df, not_found_txt


def per_product_review(review_array):
    per_product_review_array = []
    for product_review in review_array:
        tmp = product_review.split("||| ")
        per_product_review_array.append(tmp)
    return per_product_review_array

"""

# print(len(review_data))
    # print(review_data.reviewText)

    # review_data = review_data.astype(str)
    # review_data['reviewText'] = review_data['reviewText'].apply(lambda x: '||| '+ x)
    # review_data['reviewerID'] = review_data['reviewerID'].apply(lambda x: ', '+ x)
    # review_data['overall'] = review_data['overall'].apply(lambda x: ', '+ x)
    # review_data['summary'] = review_data['summary'].apply(lambda x: '||| '+ x)
    # review_data['reviewTime'] = review_data['reviewTime'].apply(lambda x: '||| '+ x)
    # newDF_groupby_product = review_data.groupby(by='asin').sum()
    # newDF_groupby_product['reviewText'] = newDF_groupby_product['reviewText'].apply(lambda x : x[4:])
    # newDF_groupby_product['summary'] = newDF_groupby_product['summary'].apply(lambda x : x[4:])
