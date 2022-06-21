#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import nltk

# colums = ['CustomerName', 'CreatedDate', 'JobNum', 'PartNum', 'Description','OperID','QtyCompleted', 'StdPrice']
# files = ['April_2021', 'July_2021', 'Oct_2021']
colums = ['CustomerName', 'CreatedDate', 'JobNum', 'PartNum', 'Description','OperID','Run Qty', 'StdPrice']
def readMerge():
    df = pd.read_csv("data/production/March_2021.csv", usecols=colums)
    df = df.append(pd.read_csv("data/production/April_2021.csv", usecols=colums))
    df = df.append(pd.read_csv("data/production/June_2021.csv", usecols=colums))
    df = df.append(pd.read_csv("data/production/July_2021.csv", usecols=colums))
    df = df.append(pd.read_csv("data/production/Sept_2021.csv", usecols=colums))
    df = df.append(pd.read_csv("data/production/Oct_2021.csv", usecols=colums))
    df = df.append(pd.read_csv("data/production/Nov_2021.csv", usecols=colums))
    return df.dropna(how='all')

def fillNA(col, m):
    col.fillna(method=m, axis=0, inplace=True) # .sample(10)

def grpMonth(df):
    df['Month'] = pd.DatetimeIndex(pd.to_datetime(df['Date'])).month
    df[['Qty','Month', 'Price']] = df[['Qty','Month','Price']].astype('float').astype(int, errors='ignore')

    grp = df[['PartNum', 'Month', 'Qty']].groupby(['PartNum','Month']).sum('Qty').    reset_index().sort_values(by='PartNum', ascending=False)
    grp['Month'] = grp['Month'].astype(int)

    return grp

# Add quantity and frequency of occurence
def addPartNum(col_List, colName):
    outDF = pd.DataFrame()
    outDF = pd.DataFrame(col_List, columns=[colName])

    return outDF

# Create List of operations for each Part#
def addOps(df):
    ops = []
    jobDesc = []
    for j in df:
        ops.append(list(productionDF['OpID'][productionDF['PartNum']==j]))
    return ops

# Add two columns for total Qty and frequency of occurence
def getQtyFreq(df_prod, df_ops):
    #     freq, qty = [], []
    df_prod['Qty'] = df_prod['Qty'].astype(float).astype('Int64', errors='ignore').fillna(0)
    # newCol = newCol

    freq = df_prod[['PartNum', 'Qty']].groupby('PartNum').count().reset_index()
    tot = df_prod[['PartNum', 'Qty']].groupby('PartNum').sum('Qty').reset_index()
    temp1 = df_ops.merge(freq, on='PartNum')
    temp = temp1.merge(tot, on='PartNum')

#     for p in partNums:
#         qty.append(sum(newCol[df['JobNum']==p]))
#         freq.append(len(newCol[df['JobNum']==p]))

    return temp

from timing import Timer

def getOps(p, prod):
    # Identifies operations for a given Part#
    f2 = ''
    foundDict, k = {}, 0
    f1 = prod['OpID'][prod['PartNum']==p]
    if len(f1)>0:
        f2 = (','.join(list(f1)).split(','))
    # opsDF['Operations'].iloc[k] = list(f2)
    # print(list(set(f2)))
    return list(set(f2))

def defectCategory(op, Dict):
    # Identifies defect Categories for given Operations, adds list to dataframe
    cats = set()
    for key in Dict.keys():
        mat = set(op).intersection(Dict[key]) # grab cat key if even one op is common btwn opsDict and catDF
        # print(key, '\t', op, '\t', mat)
        if len(mat)>0:
            cats.add(key)
    # print('Added: ', cats)
    return list(cats)

# Create the final dataframe for use in data analysis
def makeFinalDF(df, grpMonth, productionDF):
    tempDF = pd.DataFrame()
    tempDF = df[['PartNum', 'Operations', 'Category']].merge(grpMonth, on='PartNum').drop_duplicates(subset=['PartNum'])
    tempDF = tempDF.merge(productionDF[['PartNum', 'Description', 'Price']], on='PartNum').drop_duplicates(subset=['PartNum'])
    tempDF = tempDF.reset_index()
    tempDF.drop('index', axis=1, inplace=True)
    tempDF = tempDF.reindex(columns=['Month', 'PartNum', 'Description', 'Operations', 'Category', 'Qty', 'Price'])
    return tempDF

def getOp(p, prod):
    # Identifies operations for a given Part#
    f2 = ''
    foundDict, k = {}, 0
    f1 = prod['OpID'][prod['partNoRev']==p]
    if len(f1)>0:
        f2 = (','.join(list(f1)).split(','))
    # opsDF['Operations'].iloc[k] = list(f2)
    # print(list(set(f2)))
    return list(set(f2))

def removeStops(arr):
    #print(arr)
    arr = [re.sub(r'(\d*|Chin|Chiltz)', '',arr)]
    # print(arr)
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(arr)
        return vectorizer.get_feature_names_out()
    except ValueError:
        return []
    except:
        return []


def getFPY(job, mergedDF, yieldDF):
    df = mergedDF[(mergedDF['JobNum'] == job)]
    df = df[df['Dept']!='QA'] # Remove QA
    depts = df['Dept'][df['JobNum'] == job].unique() # Get all depts for the given Job#
    for dep in depts:
        dfq = df[['Date','PartNum', 'Dept','JobNum', 'Qty']][df['JobNum'] == job].query("Dept == @dep")
        qty1 = min(dfq['Qty'])
        qty2 = max(dfq['Qty'])
        part = dfq['PartNum'].unique()[0]
        date = dfq['Date'].unique()[0]
        # print("{:>10} \t {:>10} \t{:>10} \t {:>10} \t{:>15} \t\t {:.1%} ".\
        #      format(job, dep, max(dfq['Qty']), min(dfq['Qty']),  list(dfq['Qty']).count(min(dfq['Qty'])), qty1/qty2))
        yieldDF.loc[len(yieldDF.index)] = [date, job, part, dep, max(dfq['Qty']), min(dfq['Qty']),
                              list(dfq['Qty']).count(min(dfq['Qty'])), round(qty1/qty2, 1)*100]


wnetl = WordNetLemmatizer()
def getLemma(words):
    if words is not np.nan:
        # Tokenize the sentence received
        # print(words)
        sent_regex = nltk.tokenize.RegexpTokenizer("\w+")
        clean_tokens = sent_regex.tokenize(words)
        # print(clean_tokens)
        # Lemmatize the tokens
        lem_words = [wnetl.lemmatize(w, 'v') for w in clean_tokens]
        # print(lem_words)
    else:
        lem_words = None

    return lem_words