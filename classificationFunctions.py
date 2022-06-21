
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import re
import matplotlib.pyplot as plt
import datetime as dt
from scipy import stats
#
# # Import the self-created dataframes
# from DMR import dmrDF
# from production import ordersDF, fpyDF

# Analysis by Category
# Which Categories had the highest failed quantities?

# Create list of Categories in which failure occured
def getCategories(col):
    cat_temp, cat_set = set(), set()
    # apply(lambda y: re.sub(' ','_',y)))
    col.apply(lambda x:  cat_temp.add(','.join(x).strip(' ')))
#     col.apply(lambda x:  cat_temp.add(', '.join(x)))
    # print(cat_temp)
    # apply(lambda y: '' if y==None else y.split(',') ).\
    # apply(lambda x: cats.add(i.strip(' ')))

    for c in list(cat_temp):   
        # print(c)
        for i in c.split(','):
            s = i.strip(' ')            
            if len(s)>=1 and i[0]!='Not Known':                
                cat_set.add(s)            
                # print(s)
    return list(cat_set)

def categoryColumns(df, col, cols=None):
    if cols is None: 
        cats = getCategories(df[col])
    else:
        cats = cols
    # print(cats)
    for c in cats:
        try:
            # print(c, end=", ")
            if len(c)>1: df[c] = df[col].apply(lambda x: 1 if c in x else 0)
        except:            
            continue
    return df

def grpByQty(df, colums):
    grouped = df[colums].groupby(pd.DatetimeIndex(pd.to_datetime(df['Date'])).month).sum('Qty').reset_index()
    grouped.insert(loc=1, column='Month_name', value=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 
                                             'Aug', 'Sept', 'Oct', 'Nov', 'Dec'])
    return grouped

# Enter first day of each Month as date
def makeDates(col):
    dates = col.map({1:'01/01/2021', 2:'02/01/2021',3:'03/01/2021', 4:'04/01/2021',5:'05/01/2021', 6:'06/01/2021',
                       7:'07/01/2021', 8:'08/01/2021',9:'09/01/2021', 10:'10/01/2021',11:'11/01/2021', 12:'12/01/2021'})
    return dates


# # df = ncrDF[['Date', 'Category', 'Qty']]
# def annualSlope(df, colums):    
#     cats = df.drop(colums, axis=1).columns
#     trendDF=[]
    
#     for c in cats:
        
#         # df = grpByQty[['Month', 'Qty', c]]
#         df['date_ordinal'] = pd.to_datetime(df['Date']).map(dt.datetime.toordinal)
#         slope, intercept, r_value, p_value, std_err = stats.linregress(df['date_ordinal'], df[c])
#     #     print('{:>10}\t {:>5} \t{:.3f} \t {:.3f}'.format(c, 'Slope: ', slope, p_value))
#         trendDF.append([c, round(slope, 3), round(r_value, 3), round(std_err, 3), round(p_value, 3)])

#     colums = ['Category','Slope', 'r_value', 'std_err', 'p_value']
#     trend = pd.DataFrame(trendDF,index=range(len(trendDF)), columns=colums)
#     return trend

def annualSlope(df, colums):    
    cats = df.drop(colums, axis=1).columns
    data, slopes = [], ''
    for c in cats:        
        # df = grpByQty[['Month', 'Qty', c]]
        df['date_ordinal'] = pd.to_datetime(df['Date']).map(dt.datetime.toordinal)
        slope, intercept, r_value, p_value, std_err = stats.linregress(df['date_ordinal'], df[c])
        # print('{:>10}\t {:>5} \t{:.3f} \t {:.3f}'.format(c, 'Slope: ', slope, p_value))
        data.append([c, round(slope, 3), round(r_value, 3), round(std_err, 3), round(p_value, 3)])

    colums = ['Category','Slope', 'r_value', 'std_err', 'p_value']
    slopes = pd.DataFrame(data,index=range(len(data)), columns=colums)
    slopes = slopes[slopes['Category']!='date_ordinal']
    return slopes

def stripColumn(col):
    clist = []
    for c in col:
        c = re.sub(' ', '_', str(c))
        clist.append(c)
    return clist


def getSeverity(x):
    slope, pval, cdr = x[0], x[1], x[2]
    if cdr <= 1:
        sev= 4
    elif cdr <= 2 and pval >= .065:
        sev = 5
    elif 3 <= cdr <= 4 and pval >= .065:
        sev = 6
    elif 4 <= cdr >= 5 and (.05<= pval<= .065) and slope > 0:
        sev = 7
    elif 4 <= cdr >= 5 and pval<= .050 and slope > 0:
        sev = 8
    else:
        sev = 3
    return sev

def getOccurence(x, y=100):
    slope, pval= x[0], x[1]
    if pval> 0.065 and slope <= 0 :
        occurence = 3    
    elif (0.05 < pval<= .065) and (slope > 0):    
        occurence = 4
    elif pval<= 0.05 and slope > 0 and y > 95:
        occurence = 5
    elif pval<= 0.05 and slope > 0 and y <= 95:
        occurence = 6    
    else:
        occurence = 7
    return occurence

def getDetectability(x):
    slope, pval= x[0], x[1]
    if pval > 0.065 and slope <= 0 :
        detect = 3
    elif pval > 0.065 and slope > 0:
         detect = 4
    elif (0.05 < pval<= .065) and (slope >= 0):
         detect = 5
    elif (0.05 <= pval and slope >= 0):
         detect = 6
    else:
         detect = 2
    return  detect

# Function modifies risk profile based on levels of Criticality, APN and APN
def digRisk(ndf):
    df = ndf
    for i in range(df.shape[0]):
        # print('Criticality: ', df.iloc[i][0], '\tRPN: ', df.iloc[i][2], '\tRisk: ', df.iloc[i][3])
        if (((df.iloc[i][0] >= 20) or (df.iloc[i][1] >= 20)) & (df.iloc[i][2] < 120)):            
            df.iloc[i, df.columns.get_loc('Risk')] = 'High'
            # print(df.iloc[i][0],'\t', df.iloc[i][1], '\t', df.iloc[i][3])
        else:
            continue   
    
    return df

def riskProfile(df):
    df['Severity'] = df[['Slope', 'p_value', 'cdr']].apply(lambda x: getSeverity(x), axis=1)
    df['Occurence'] = df[['Slope', 'p_value']].apply(lambda x: getOccurence(x), axis=1)
    df['Detectability'] = df[['Slope', 'p_value']].apply(lambda x: getDetectability(x), axis=1)
    df['Criticality'] = df['Severity']*df['Occurence']
    df['APN'] = df['Severity']*df['Detectability']
    df['RPN'] = df['Severity']*df['Occurence']*df['Detectability']
    df['Risk'] = pd.cut(df['RPN'], bins=[0, 79, 95, 120, 500], include_lowest=True, labels=['Low', 'Medium', 'High', 'Very High'])
    df[['Criticality', 'APN', 'RPN', 'Risk']] = digRisk(df[['Criticality', 'APN', 'RPN', 'Risk']])
    return df

def findRecord(recordDF, recordCol, searchCol):
    found = list(set(recordCol).intersection(searchCol))
    df = recordDF[recordCol.isin(found)]
    return df

