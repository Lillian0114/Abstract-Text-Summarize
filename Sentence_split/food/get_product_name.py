import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from tqdm import tqdm
import gzip
import numpy as np
from amazoncaptcha import AmazonCaptcha
from selenium.common.exceptions import NoSuchElementException
from selenium import webdriver
import json


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

def loadsmall(file_path):
    with open(file_path, 'r') as file:
        review_smalldata = pd.read_csv(file)
    return review_smalldata

# load dataset
def load_asin(file_path):
    with open(file_path, 'r') as file:
        group_review = pd.read_csv(file)

    asin_array = group_review['asin'].values
        
    return asin_array

def load_PreDataSave(product_name_path, new_amazon_path, not_found_path):
    with open(product_name_path, 'r',encoding="utf-8") as file:
        products_df = pd.read_csv(file)
    with open(new_amazon_path, 'r', encoding="utf-8") as file:
        new_df = pd.read_csv(file)
    with open(not_found_path, 'r',encoding="utf-8") as file:
        file_content = file.read()
        not_found_txt = file_content.split(", ")
        
    return products_df, new_df, not_found_txt

def main():
    cwd = os.getcwd() #返回當前目錄
    category = "cellphone_smalldataset.csv"
    # category = "Grocery_and_Gourmet_Food"
    # txtPath = os.path.join(cwd,'dataset',f'{category}_reviews.json.gz') 
    # review_data = getDF(txtPath) # 151254
    # review_data.to_csv('originalFood.csv')
    txtPath = os.path.join(cwd,'dataset',f'{category}') 
    review_data = loadsmall(txtPath) # 151254
    tmp = review_data['asin'].values
    asin_array = np.unique(tmp) # 8713
    # print(len(asin_array))
    products_df = pd.DataFrame()
    new_df = pd.DataFrame()
    cannot_found = pd.DataFrame()
    # products_df, new_df, cannot_found = load_PreDataSave("product_name.csv","new0201_amazondataset.csv","file.txt")
    for asin_num in tqdm(asin_array):
        url = "http://www.amazon.co.uk/dp/"+str(asin_num)
        driver = webdriver.Chrome(executable_path='./chromedriver')
        driver.get(url)

        try:
            link = driver.find_element_by_xpath("/html/body/div/div[1]/div[3]/div/div/form/div[1]/div/div/div[1]/img").get_attribute("src")
            # print(link)
            captcha = AmazonCaptcha.fromlink(link)
            solution = captcha.solve()
            captcha_textbox = driver.find_element_by_xpath("/html/body/div/div[1]/div[3]/div/div/form/div[1]/div/div/div[2]/input")
            captcha_textbox.send_keys(solution)
            driver.find_element_by_xpath("/html/body/div/div[1]/div[3]/div/div/form/div[2]/div/span/span/button").click()
        except:
            pass
        try:
            driver.find_element_by_class_name("sp-cc-buttons")
            driver.find_element_by_id("sp-cc-accept").click()
            driver.implicitly_wait(10)

            title = str(driver.find_element_by_id("productTitle").text)
            new_row = {'asin_num':asin_num, 'product_name':title}
            products_df = products_df.append(new_row, ignore_index=True)
            print("PRODUCT TITLE: ", title)
            driver.close()
        except NoSuchElementException: 
            cannot_found.append(asin_num)
            print("exception handled")
            driver.close()
        except Exception as e:
            print("PRODUCT NOT ACCESSIBLE")
            print("Error in Product Access", e)
            driver.close()
    
    print(len(cannot_found))
    print(len(products_df))
    
    products_df.to_csv("product_name.csv",index=False)
    new_df.append(review_data[review_data.asin.isin(cannot_found) == False])
    new_df.to_csv('new0201_amazondataset.csv',index=False)

if __name__ == '__main__':
    main()



    # new_df = review_data[review_data.asin.isin(cannot_found) == False]
    # new_df2 = review_data[new_df.asin.isin(products_df['asin_num']) == False]
    # new_df2.to_csv("abcde.csv",index=False)
    # saving the dataframe
    # with open("notfound_product.txt", "w") as output:
    #     output.write(str(cannot_found))
"""
def main():
    cwd = os.getcwd() #返回當前目錄
    category = "Grocery_and_Gourmet_Food"
    txtPath = os.path.join(cwd,'dataset',f'{category}_reviews.json.gz') 
    review_data = getDF(txtPath) # 151254
    # asin_array = load_asin(txtPath)
    # review_data = load_tmp("rrrrr.csv")
    tmp = review_data['asin'].values
    asin_array = np.unique(tmp)
    products_df = pd.DataFrame(columns = ["asin_num", "product_name"])
    cannot_found = []
    for asin_num in tqdm(asin_array):
        url = "http://www.amazon.co.uk/dp/"+str(asin_num)
        # header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'}
        header = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
        page = requests.get(url, headers=header)
        
        if page.status_code == 404:
            cannot_found.append(asin_num)
        else:
            soup = BeautifulSoup(page.content,"lxml")
            try:
                pTitle = soup.find(id='productTitle').get_text()
                new_row = {'asin_num':asin_num, 'product_name':pTitle}
                products_df = products_df.append(new_row, ignore_index=True)
            except AttributeError:
                # print ("asin-num: ",asin_num)
                # break
                captcha = AmazonCaptcha.fromlink('https://www.amazon.com/errors/validateCaptcha')
                solution = captcha.solve()
            
    
    print(len(cannot_found))
    print(len(products_df))
    
    products_df.to_csv("product_name.csv",index=False)
    new_df = review_data[review_data.asin.isin(cannot_found) == False]
    new_df.to_csv('new0201_amazondataset.csv',index=False)
    new_df2 = review_data[new_df.asin.isin(products_df['asin_num']) == False]
    new_df2.to_csv("abcde.csv",index=False)
    # saving the dataframe
    with open("file.txt", "w") as output:
        output.write(str(cannot_found))
    # a_file = open("cannot_found_name.txt", "w")
    # for row in cannot_found:
    #     np.savetxt(a_file, row, delimiter=',')
    # a_file.close()
    
"""