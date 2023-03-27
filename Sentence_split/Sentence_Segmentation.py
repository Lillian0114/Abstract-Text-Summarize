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

spell  = SpellChecker()
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

# load dataset
def load_Preprocess_review(file_path):
    with open(file_path, 'r', encoding="utf-8",errors='ignore') as file:
        group_review = pd.read_csv(file)
    return group_review

def load_coexist_pkl(file_path):
    with open(file_path, 'rb') as f:
        coexist_dict = pickle.load(f)
    return coexist_dict

# stopwords
def load_stopwords(path):
    stop_words = []
    f = open(path, "r")
    for line in f:
        line = line.strip('\n').strip()
        stop_words.append(line)
    return stop_words

def cleaning_word(word, stopwords, lemma = True):
    # word, pos = wordpos
    # productName = productName.translate(str.maketrans('', '', string.punctuation)).lower().split()
    word = re.sub(r'https?:\/\/.*[\r\n]*', '', word, flags=re.MULTILINE)
    # html entity decode
    word = html.unescape(word)
    # remove the punctuation
    word = word.translate(str.maketrans('', '', string.punctuation))
    # remove tokens that don't contain letters or numbers  
    word = re.sub(r"[^A-Za-z0-9`]", " ", word)
    if spell.correction(word) is not None:
        word = spell.correction(word)

    if not str.isalpha(word):
        word = ""
    if lemma and word!="":
        word = lemmatizer.lemmatize(word,'n')
    
    if word in stopwords:
        word = ""

    return word

# def CalFrequentwords(dataframeReview, stopwords):
def CalFrequentwords(dataframeReview):
    JVN_Freqwords = {}
    N_Freqwords = {}
    for idx, review in tqdm(dataframeReview.iterrows()):
        # freqtmp = review['pos_noun_verb_adj'].replace("[","").replace("]","").replace("'","").split(",")
        # freqtmp = [word.split(',')[0].replace(" ", "") for word in review['pos_noun_verb_adj'].replace("[","").replace("]","").replace("(","").replace("'","").split("),")]
        n_frequtmp =[word.split(',')[0].replace(" ", "") for word in review['pos_noun'].replace("[","").replace("]","").replace("(","").replace("'","").split("),")]
        for word in n_frequtmp:
            # print(word)
            # word = cleaning_word(word, stopwords)
            # # print(word)
            # if word == "" or word == " ":
            #     continue
            if word.strip() not in N_Freqwords:
                N_Freqwords[word.strip()] = 1
            else:
                N_Freqwords[word.strip()] += 1 
        # for word in freqtmp:
        #     word = cleaning_word(word, stopwords, lemma =False)
        #     if word == "" or word == " ":
        #         continue
        #     if word.strip() not in JVN_Freqwords:
        #         JVN_Freqwords[word.strip()] = 1
        #     else:
        #         JVN_Freqwords[word.strip()] += 1 
    return JVN_Freqwords,N_Freqwords

def dict_clean(Freqwords,stopwords):
    removewords = []
    other_dict = {}
    for word, value in tqdm(Freqwords.items()) :
        newword = cleaning_word(word,stopwords)
        if newword == "" or newword==" ":
            # del Freqwords[word]
            removewords.append(word)
        else:
            if newword in Freqwords and newword!= word:
                Freqwords[newword] += value
                removewords.append(word)
            elif newword not in Freqwords and newword!= word:
                other_dict[newword] = value
                removewords.append(word)
    for remove in removewords:
        del Freqwords[remove]
    Freqwords.update(other_dict)
    # print(removewords)
    return Freqwords

# ================== coexist =================
def build_occurrence_matrix(review_data, FreqwordArray, coexist_threshold):
    # Build co-occurrence matrix
    cooc_mat = defaultdict(int)
    # co_matrix = np.zeros((len(FreqwordArray), len(FreqwordArray)))
    for idx, perreview in tqdm(review_data.iterrows()):
        words = perreview['review_clean'].split()
        for i in range(len(words)):
            if words[i] not in FreqwordArray:
                continue
            for j in range(max(i - coexist_threshold, 0), min(i + coexist_threshold + 1, len(words))):
                if i != j and words[j] in FreqwordArray:
                    cooc_mat[(words[i], words[j])] += 1
    return cooc_mat

def cal_confidence(co_occurence,FreqWordArray, word2id):
    # Determine the size of the matrix
    num_rows = len(set(row[0] for row in co_occurence.keys()))
    num_cols = len(set(row[1] for row in co_occurence.keys()))
    # num_rows = len(FreqWordArray)
    # num_cols = len(FreqWordArray)
    print(num_rows)
    # Initialize the column sums to zero
    col_sums = [0] * num_cols
    # Sum the values for each column index
    for col_idx in tqdm(range(num_cols)):
        for row_idx in range(num_rows):
            key = (list(word2id.keys())[col_idx],list(word2id.keys())[row_idx])
            col_sums[col_idx] += co_occurence[key]
    print("column sume done")
    print(len(col_sums))

    confidence_matrix = np.zeros((len(FreqWordArray), len(FreqWordArray)))
    for i in tqdm(range(num_cols)):
        for j in range(num_rows):
            if i == j:
                confidence_matrix[i,j] = 1.0
            else:
                key =  (list(word2id.keys())[i],list(word2id.keys())[j])
                # print(key)
                count = co_occurence[key]
                if count!= 0:
                    c1 = count/ col_sums[i]
                    c2 = count / col_sums[j]
                    confidence_value = max(c1, c2)
                else:
                    confidence_value = 0
                confidence_matrix[i,j] = confidence_value
    return confidence_matrix

def segment_sentence(reviewdataframe, FreqWordArray, word2id,
                     confidence_matrix, threshold):
    # split_sentence = []
    for idx,df in tqdm(reviewdataframe.iterrows()):
        firstfrequent = True
        previosFrequent = None
        thisreview_split_sentence = []
        # review = review.translate(str.maketrans('', '', string.punctuation))
        review = df['review_clean']
        sentence = review
        # print(review.split())
        # print(len(review.split()))
        for word in review.split():
            if word in FreqWordArray and firstfrequent:
                previosFrequent = word
                firstfrequent = False
            elif word in FreqWordArray and firstfrequent==False and word!=previosFrequent:
                w1 = word2id[word]
                w2 = word2id[previosFrequent]
                if w1 > len(confidence_matrix) or w2 > len(confidence_matrix):
                    continue
                if confidence_matrix[w1,w2] < threshold:
                    thisreview_split_sentence.append(sentence[:sentence.index(word)].strip())
                    sentence = sentence[sentence.index(word):]
                previosFrequent = word
        # Add the final part of the sentence to the split sentence list
        thisreview_split_sentence.append(sentence)
        # print(thisreview_split_sentence)
        # print(idx," ???????????")
        # split_sentence.append(thisreview_split_sentence)
        reviewdataframe.at[idx,'segmentation'] = thisreview_split_sentence
    return reviewdataframe

def no_clean_segment_df(reviewdataframe, FreqWordArray, word2id,
                     confidence_matrix, threshold):
    # split_sentence = []
    for idx,df in tqdm(reviewdataframe.iterrows()):
        firstfrequent = True
        previosFrequent = None
        thisreview_split_sentence = []
        # review = review.translate(str.maketrans('', '', string.punctuation))
        review = df['body']
        sentence = review
        for word in review.split():
            cleanword = word.translate(str.maketrans('', '', string.punctuation))
            if cleanword in FreqWordArray and firstfrequent:
                previosFrequent = cleanword
                firstfrequent = False
            elif cleanword in FreqWordArray and firstfrequent==False and cleanword!=previosFrequent:
                w1 = word2id[cleanword]
                w2 = word2id[previosFrequent]
                if w1 > len(confidence_matrix) or w2 > len(confidence_matrix):
                    continue
                if confidence_matrix[w1,w2] < threshold:
                    thisreview_split_sentence.append(sentence[:sentence.index(word)].strip())
                    sentence = sentence[sentence.index(word):]
                previosFrequent = cleanword
        # Add the final part of the sentence to the split sentence list
        thisreview_split_sentence.append(sentence)
        reviewdataframe.at[idx,'notclean_segmentation'] = thisreview_split_sentence
    return reviewdataframe

def main():
    cwd = os.getcwd() 
    #====== hyperparameters =======
    confidence_threshold = 0.04
    word_coexist_threshold = 5
    #==============================
    # filename = "tmpppp.csv"
    # txtPath = os.path.join(cwd,'dataset',filename) 
    # filename = "phone0302_pos.csv"
    stopwordname = "stopwords.txt"
    txtPath3 = os.path.join(cwd,'dataset',f'{stopwordname}') 
    stop_words = load_stopwords(txtPath3)
    custom_stop = ['another','anybody',"anyhow",'anyone','anything','anyway','anyways','anywhere','phone']
    stop_words = stop_words
    # stop_words.remove('not')
    stop_words.extend(custom_stop) 

    filename = "phone0325_Npos.csv"
    txtPath = os.path.join(cwd,'dataset','cellphone',filename) 
    review_data = load_Preprocess_review(txtPath) 
    # print(len(review_data))
    review_data = review_data[review_data['review_clean'].notna()]
    review_data = review_data[review_data['pos_noun_verb_adj'].notna()]
    review_data = review_data[review_data['pos_noun_verb_adj'] != '[]']
    review_data = review_data[review_data['pos_noun'].notna()]
    review_data = review_data[review_data['pos_noun'] != '[]']
    keywords = review_data["brand"].astype(str).apply(lambda x: x.lower()).unique().tolist()
    stop_words.extend(keywords) 
    review_data['segmentation'] = review_data['review_clean']
    review_data['notclean_segmentation'] = review_data['body']
    print(len(review_data))
    # # review_len = len(review_data)
    # # keywords = review_data["asin"].unique().tolist()
    # # print(len(keywords))
    # # print(keywords)

    # # #====== CoexitWords =======
    # # # df_pnone_bigReviews = pd.DataFrame(review_data[['asin','review_clean']].groupby(['asin'])['review_clean'].progress_apply(list).reset_index(name="groupbyasin"))
    # JVN_FreqWordsDict, N_FreqWordsDict = CalFrequentwords(review_data)
    # # print(N_FreqWordsDict)
    # print(len(N_FreqWordsDict))
    # N_FreqWordsDict = dict_clean(N_FreqWordsDict,stop_words)
    # for k in list(N_FreqWordsDict.keys()):
    #     if N_FreqWordsDict[k] < 2:
    #         del N_FreqWordsDict[k]
    # # print(N_FreqWordsDict)
    # # for k in list(JVN_FreqWordsDict.keys()):
    # #     if JVN_FreqWordsDict[k] < 2:
    # #         del JVN_FreqWordsDict[k]

    # with open('FreqWordsDict.pkl', 'wb') as fp:
    #     pickle.dump(N_FreqWordsDict, fp)
    #     print('dictionary saved successfully to file')
    
    dict_path = "FreqWordsDict.pkl"
    N_FreqWordsDict = load_coexist_pkl(dict_path)
    print('load dict successfully')
    # print(sorted(N_FreqWordsDict.items(), key=lambda x:x[1]))
    removewords = ['use','good','great','get','work','one','new','go',
                   'would','buy','love','like','really','make','come',
                   'even','much','someone','thats','let','also','still',
                   'aswell','well','lot','nothing','anymore','many','sooo',
                   'soooo','sooooo','soooooo','everything','ive','wife','girl',
                   'man','people','friend','etc','feel','con','someone','let','talk',
                   'havent','other','others','pa','try','waste','al','ok','buying',
                   'kid','adult','say','easy','use','hear','person','talk','fine',
                   'whenever','oh','keep','yesterday','sombody','child','children',
                   'long','better','want','dont','thing','nice','bad','otherwise',
                   'shes','yr','yet','ha','fun','different','pay','okay','friendly',
                   'e','theyre','fi','le','best','year','hour','mininute','two','pretty',
                   'cool','way','hard','press','la','think','ever','everyone','wa','excellent',
                   'mom','son','boy','boyfriend','girl','girlfriend','youre','youll','youve']
    for k in list(N_FreqWordsDict.keys()):
        if N_FreqWordsDict[k] < 51:
            del N_FreqWordsDict[k]
        elif k in removewords:
            del N_FreqWordsDict[k]
    # print(N_FreqWordsDict)
    # Build vocabulary
    FreqwordArray = list(N_FreqWordsDict.keys())
    print(sorted(N_FreqWordsDict.items(), key=lambda x:x[1]))
    print(len(FreqwordArray))

    word2id = {}
    for i, word in enumerate(FreqwordArray):
        word2id[word] = i
    # print(word2id)
    
    Noun_coexitence_dict =  build_occurrence_matrix(review_data, FreqwordArray, word_coexist_threshold)
    print("build co_occurence matrix done")
    print("共現性長度",len(Noun_coexitence_dict))
    # # Print heatmap for top words by frequency
    # top_words = sorted(FreqwordArray, key=lambda w: sum([Noun_coexitence_dict[(w, w2)] for w2 in FreqwordArray]), reverse=True)[:15]
    # print(top_words)
    # cooc_mat_arr = np.zeros((len(top_words), len(top_words)))
    # for i, w1 in enumerate(top_words):
    #     for j, w2 in enumerate(top_words):
    #         cooc_mat_arr[i, j] = Noun_coexitence_dict[(w1, w2)]
    # plt.imshow(cooc_mat_arr, cmap='YlOrRd', interpolation='nearest')
    # plt.xticks(np.arange(len(top_words)), top_words, rotation=60)
    # plt.yticks(np.arange(len(top_words)), top_words)
    # plt.tick_params(top=True, bottom=False,
    #             labeltop=True, labelbottom=False)
    # # for i in range(len(top_words)):
    # #     for j in range(len(top_words)):
    # #         plt.text(j, i, int(cooc_mat_arr[i, j]), ha="center", va="center", color="black")
    # plt.show()

    #====== Sentence Segement =======
    confidence_matrix = cal_confidence(Noun_coexitence_dict,FreqwordArray, word2id)
    print("build confidence matrix done")
    segment_dataframe = segment_sentence(review_data, FreqwordArray, word2id,
                                        confidence_matrix, confidence_threshold)
    segment_dataframe = no_clean_segment_df(segment_dataframe, FreqwordArray, word2id,
                                        confidence_matrix, confidence_threshold)
    print("build segment dataframe done")
    segment_dataframe.to_csv('0325phone_segment.csv',index=None)
    # segment_dataframe.to_csv('0325tmp_segment.csv',index=None)

    # with open('coexist.pkl', 'wb') as fp:
    #     pickle.dump(coexitence_dict, fp)
    #     print('dictionary saved successfully to file')
    # #====== Sentence Segement =======
    # dict_path = "coexist.pkl"
    # coexitence_dict = load_coexist_pkl(dict_path)
    # # print(pd.Series(list(coexitence_dict.values())).describe())
    # # support_dict =  {key: val/review_len for key, val in coexitence_dict.items()}
    # new_coexitence_dict = {k: v for k, v in coexitence_dict.items() if v > 2}
    # # print(pd.Series(list(new_coexitence_dict.values())).describe())

    # #======= Sentence_Segment =======
    # segement_array = calculate_segment(review_data, FreqWordsDict,new_coexitence_dict, confidence_threshold)
    # # print(segement_array)
    # segement_array.to_csv('phone_segement.csv',index=None)
    
    

if __name__ == '__main__':
    main()




# def collect_zipfs_law_metrics(review, fd):
#     for token in review:
#         fd.update([token])

# def filter_words(review, WordTermFreq):
#     new_review = []
#     for word in review.split():
#         # word = word.strip()
#         if word in WordTermFreq['word'].values.tolist():
#             new_review.append(word)
#     return new_review

# def apply_arm(transactions):
#     return list(apriori(transactions, min_support = 1/len(transactions), min_confidence = 1, min_lift = len(transactions), max_length = 4)) 

# def get_important_nouns(arms):
#     imp_nns = []
#     # print(arms.keys())
#     if "items" in pd.DataFrame(arms).keys():
#         results = list(pd.DataFrame(arms)['items'])
#         for result in results:
#             if len(list(result)) > 4:
#                 imp_nns = imp_nns + list(list(result))
#         if(len(imp_nns)==0):
#             for result in results:
#                 if len(list(result)) > 3:
#                     imp_nns = imp_nns + list(list(result))            
#         return list(set(imp_nns))
#     return list(set(imp_nns))

# imp_nns_df = imp_nns_df.assign(num_of_imp_nouns = imp_nns_df['imp_nns'].progress_apply(lambda imp_nouns:len(imp_nouns)))
# imp_nns_df.head()



#### main
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

"""
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
"""