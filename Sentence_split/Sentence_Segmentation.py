import nltk
from nltk.probability import FreqDist
import os
import pandas as pd
import numpy as np
from tqdm import tqdm, tqdm_pandas
import string
import numpy as np, codecs, json, pickle, sys
from apyori import apriori
import gzip
tqdm.pandas()

# load dataset
def load_Preprocess_review(file_path):
    with open(file_path, 'r', encoding="utf-8",errors='ignore') as file:
        group_review = pd.read_csv(file)
    return group_review

def load_coexist_pkl(file_path):
    with open(file_path, 'rb') as f:
        coexist_dict = pickle.load(f)
    return coexist_dict


def collect_zipfs_law_metrics(review, fd):
    for token in review:
        fd.update([token])

def filter_words(review, WordTermFreq):
    new_review = []
    for word in review.split():
        # word = word.strip()
        if word in WordTermFreq['word'].values.tolist():
            new_review.append(word)
    return new_review

def apply_arm(transactions):
    return list(apriori(transactions, min_support = 1/len(transactions), min_confidence = 1, min_lift = len(transactions), max_length = 4)) 

def get_important_nouns(arms):
    imp_nns = []
    # print(arms.keys())
    if "items" in pd.DataFrame(arms).keys():
        results = list(pd.DataFrame(arms)['items'])
        for result in results:
            if len(list(result)) > 4:
                imp_nns = imp_nns + list(list(result))
        if(len(imp_nns)==0):
            for result in results:
                if len(list(result)) > 3:
                    imp_nns = imp_nns + list(list(result))            
        return list(set(imp_nns))
    return list(set(imp_nns))


def CalFrequentwords(dataframeReview):
    Freqwords = {}
    for idx, review in tqdm(dataframeReview.iterrows()):
        freqtmp = review['pos_noun_verb_adj'].replace("[","").replace("]","").replace("'","").split(",")
        for word in freqtmp:
            if word.strip() not in Freqwords:
                Freqwords[word.strip()] = 1
            else:
                Freqwords[word.strip()] += 1 
        # print(Freqwords)
        # break
    return Freqwords

# ================== coexist =================
def word_coexist(first, second, sentence, word_distance_threshold, coexistence):
    count = 0
    # if first in sentence and second in sentence:
    # tmp = np.array( sentence.split() )
    tmp = np.array(sentence)
    first_occur = (np.where(tmp == first)[0]+1)
    second_occur = (np.where(tmp == second)[0]+1)
    max_abs_len = max(len(first_occur), len(second_occur))
    # print(first_occur," ",second_occur)
    if len(first_occur)!= max_abs_len:
        while len(first_occur)!= max_abs_len:
            first_occur = np.append(first_occur, first_occur[len(first_occur) - 1])
    elif len(second_occur)!= max_abs_len:
        while len(second_occur)!= max_abs_len:
            second_occur = np.append(second_occur, second_occur[len(second_occur) - 1])
    word_distance = abs( first_occur -  second_occur)
    # print("word_dist: ", word_distance)
    for i in range(len(word_distance)):
        if word_distance[i] <= word_distance_threshold:
            count += 1
    multikey1 = (first, second)
    multikey2 = (second, first)
    if multikey1 in coexistence:
        coexistence[multikey1] += count
    elif multikey2 in coexistence:
        coexistence[multikey2] += count
    else:
        if count!=0:
            coexistence[multikey1] = count
        else:
            coexistence = coexistence
    
    return coexistence

def calulate_coexistence(review_data, FreqwordArray, word_coexist_threshold):
    coexitence_dict = {}
    for i in tqdm( range(len(FreqwordArray))  ):
        first = FreqwordArray[i]
        # for j in range(1,len(FreqwordArray)):
        #     second = FreqwordArray[j]
        for idx, perreview in review_data.iterrows():
            sentence = perreview['review_clean'].split()
            if first not in sentence:
                continue
            for second in sentence:
                if second not in FreqwordArray or first == second:
                    continue
                print("cal coexit")
                coexitence_dict = word_coexist(first, second, sentence, word_coexist_threshold, coexitence_dict)
                # break
            # break
        # break
    return coexitence_dict

# ========== Support ===========


# ========== Sentence Segment ===========
def confidence(word, previosFrequent,coexist_dict,wordTermFreq):
    count = 0
    if (word, previosFrequent) in coexist_dict:
        count = coexist_dict[(word, previosFrequent)]
    elif (previosFrequent, word) in coexist_dict:
        count = coexist_dict[(previosFrequent,word)]
    c1 = count / wordTermFreq[word]
    c2 = count / wordTermFreq[previosFrequent]
    confidence_value = max(c1, c2)
    return confidence_value

def splitsentence(sentence, element):
    a = sentence.split(element+" ")
    # sentence_splitment = a[0]+ element+ " "
    sentence_splitment = a[0]
    newreview = ""
    if len(a)>1:
        for idx,other in enumerate(a[1:]):
            if idx!=len(a[1:]):
                newreview += element+ " "
            if idx == len(a[1:])-1 and other =="":
                newreview += element
            else:
                newreview += other
    else:
        newreview = sentence
    # for idx, sentence in enumerate(a):
    #     if idx != len(a):
    #         sentence_splitment = sentence + element
    return sentence_splitment, newreview

def calculate_segment(reviewdataframe,Freqwords, coexist_dict,confidence_threshold):
    firstfrequent = True
    previosFrequent = None
    # print(Freqwords)
    # for review in tqdm(allreview.split(".||| ")):
    for idx,df in tqdm(reviewdataframe.iterrows()):
        tmparray = []
        # review = review.translate(str.maketrans('', '', string.punctuation))
        review = df['review_clean']
        newreview = review
        # print(review.split())
        # print(len(review.split()))
        for idex, word in enumerate(review.split()):
            if word in list(Freqwords.keys()):
                # print(word)
                if not firstfrequent:
                    # print("?")
                    # print(idex)
                    if idex == (len(review.split())-1):
                        # print(newreview)
                        # print("????????")
                        tmparray.append(newreview)
                    elif previosFrequent in newreview.split() and previosFrequent!=word:
                        confidence_value = confidence(word, previosFrequent,coexist_dict,Freqwords)
                        if confidence_value < confidence_threshold:
                            sentence_splitment, newreview = splitsentence(newreview, word)
                            tmparray.append(sentence_splitment)
                else:
                    firstfrequent = False
                previosFrequent = word
        if len(tmparray) == 0:
            tmparray.append(review)
        # segement_array.append(tmparray)
        firstfrequent = True
        previosFrequent = None
        reviewdataframe.at[idx,'segement'] = tmparray
    # return segement_array
    return reviewdataframe

def main():
    cwd = os.getcwd() 
    #====== hyperparameters =======
    confidence_threshold = 0.2
    word_coexist_threshold = 5
    #==============================
    filename = "phone0302_pos.csv"
    txtPath = os.path.join(cwd,'dataset','cellphone',filename) 
    # filename = "tmpppp.csv"
    # txtPath = os.path.join(cwd,'dataset',filename) 
    review_data = load_Preprocess_review(txtPath) 
    # review_data['wordCountBefore'] = 0
    # review_data['wordCountAfter'] = 0
    # review_data['filteredText'] = ""
    review_data = review_data[review_data['review_clean'].notna()]
    review_data = review_data[review_data['pos_noun_verb_adj'].notna()]
    review_data = review_data[review_data['pos_noun_verb_adj'] != '[]']
    review_data['segement'] = review_data['review_clean']
    # review_len = len(review_data)
    # keywords = review_data["asin"].unique().tolist()
    # print(len(keywords))
    # print(keywords)

    # #======= TermFreq =======
    # fd = FreqDist()
    # for idx, review in tqdm(review_data.iterrows()):
    #     collect_zipfs_law_metrics(review['review_clean'].split(), fd)
    # words = []
    # freqs = []
    # for rank, word in enumerate(fd):
    #     words.append(word)
    #     freqs.append(fd[word])
    # frequencies = {'word': words, 'frequency':freqs}
    # WordTermFreq = pd.DataFrame(frequencies)
    # WordTermFreq = WordTermFreq.sort_values(['frequency'], ascending=[False])
    # WordTermFreq = WordTermFreq.reset_index()
    # WordTermFreq = WordTermFreq.drop(columns=['index'])
    # WordTermFreq['word'] = WordTermFreq['word'].progress_apply(lambda word: word.replace(" ",""))
    # WordTermFreq = WordTermFreq.reset_index()
    # WordTermFreq = WordTermFreq.drop(columns=['index'])
    # print(WordTermFreq)
    # #======= Word2Id =======
    # filtered_dict = WordTermFreq['word'].to_dict()
    # inv_filtered_dict = {v: k for k, v in filtered_dict.items()}
    # # print(inv_filtered_dict)
    # #=============================

    #====== CoexitWords =======
    # # df_pnone_bigReviews = pd.DataFrame(review_data[['asin','review_clean']].groupby(['asin'])['review_clean'].progress_apply(list).reset_index(name="groupbyasin"))
    # # print(df_pnone_bigReviews)
    FreqWordsDict = CalFrequentwords(review_data)
    # print(FreqWordsDict)
    # print(sorted(FreqWordsDict.items(), key=lambda x:x[1]))
    # FreqwordArray = list(FreqWordsDict.keys())
    # coexitence_dict =  calulate_coexistence(review_data, FreqwordArray, word_coexist_threshold)
    # print(sorted(coexitence_dict.items(), key=lambda x:x[1]))
    # with open('coexist.pkl', 'wb') as fp:
    #     pickle.dump(coexitence_dict, fp)
    #     print('dictionary saved successfully to file')

    dict_path = "coexist.pkl"
    coexitence_dict = load_coexist_pkl(dict_path)
    # print(pd.Series(list(coexitence_dict.values())).describe())
    # support_dict =  {key: val/review_len for key, val in coexitence_dict.items()}
    new_coexitence_dict = {k: v for k, v in coexitence_dict.items() if v > 2}
    # print(pd.Series(list(new_coexitence_dict.values())).describe())

    #======= Sentence_Segment =======
    segement_array = calculate_segment(review_data, FreqWordsDict,new_coexitence_dict, confidence_threshold)
    # print(segement_array)
    segement_array.to_csv('phone_segement.csv',index=None)
    
    # print(WordTermFreq['frequency'].describe()) #20005
    # FilterWordTermFreq = WordTermFreq.loc[WordTermFreq['frequency'] > 2]
    # print(WordTermFreq.loc[WordTermFreq['frequency'] > 2].describe())
    # print(WordTermFreq['word'].loc[WordTermFreq['frequency'] > 2].count())
    # # Use threshold for 75 quantile
    # # print(FilterWordTermFreq['frequency'].describe())
    # final_dic = FilterWordTermFreq.loc[WordTermFreq['frequency'] < 20]
    # print(len(final_dic))
    
    # for idx, review in tqdm(review_data.iterrows()):
    #     review_data.at[idx,'wordCountBefore'] = len(review['pos_noun'].split())

    # for idx, review in tqdm(review_data.iterrows()):
    #     review_data.at[idx,'filteredText'] = filter_words(review['pos_noun'],final_dic)
    
    # for idx, review in tqdm(review_data.iterrows()):
    #     review_data.at[idx,'wordCountAfter'] = len(review['filteredText'])

    # remaining = 1 - review_data['wordCountAfter'].sum() / review_data['wordCountBefore'].sum()
    # print("Average noun reduction achieved:" + str(remaining*100) + "%")
    # # print(review_data)

    # df_pnone_bigReviews = pd.DataFrame(review_data[['asin','filteredText']].groupby(['asin'])['filteredText'].progress_apply(list))
    # df_pnone_bigReviews = df_pnone_bigReviews.reset_index()
    # df_pnone_bigReviews = df_pnone_bigReviews.assign(transactions = df_pnone_bigReviews['filteredText'].progress_apply(lambda reviews_lis:len(reviews_lis)))
    # # print(df_pnone_bigReviews.head(16))
    # # df_pnone_bigReviews.to_csv("tmpp.csv")

    # phone_with_arm = df_pnone_bigReviews.assign(arm = df_pnone_bigReviews['filteredText'].progress_apply(lambda list_of_reviews:apply_arm(list_of_reviews)))
    # imp_nns_df = phone_with_arm.assign(imp_nns = phone_with_arm['arm']
    #                                .progress_apply(lambda arms:get_important_nouns(arms)))
    # print(imp_nns_df.head())
    # print(phone_with_arm.head())
    
    
    

    

if __name__ == '__main__':
    main()

# imp_nns_df = imp_nns_df.assign(num_of_imp_nouns = imp_nns_df['imp_nns'].progress_apply(lambda imp_nouns:len(imp_nouns)))
# imp_nns_df.head()