#!/usr/bin/env python
# coding: utf-8

# In[38]:


import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

# Import functions to extract and process production data
from productionFunctions import *

colums = ['CustomerName', 'CreatedDate', 'JobNum', 'PartNum', 'Description', 'OperID', 'Run Qty', 'StdPrice']
files = ['April_2021', 'July_2021', 'Oct_2021']

productionDF = readMerge()
productionDF.columns = ['Customer', 'JobNum', 'Date', 'PartNum', 'Description', 'OpID', 'Qty', 'Price']

cols = ['Customer', 'PartNum', 'Price', 'Qty', 'OpID']
for c in cols:
    fillNA(productionDF[c], 'ffill')
    fillNA(productionDF[c], 'bfill')

# Group operations by Part#

# List of all Part numbers
partNums = productionDF['PartNum'].dropna().unique()

# pd.to_datetime(productionDF['Date'])

# Remove commas from Qty numbers
productionDF['Qty'] = productionDF['Qty'].apply(lambda x: re.sub(',', '', str(x)))
productionDF['Price'] = productionDF['Price'].apply(lambda x: re.sub(',', '', str(x).strip('$, '))).apply(
    lambda x: re.sub('-', '0', str(x)))

prodDF = productionDF
# Group data by Month
grpByMonth = grpMonth(productionDF)

# Create dataframe with one column
opsDF = addPartNum(partNums, 'PartNum')

# Create dataframe of operations required for each defect Category, begi with dictionary of Category and key_words<br>
# used in Operations that point to the Category
opsDict = {
    'GD&T': ['Cell4_GR', 'Cell1VF3', 'Cell1VF6', 'Cell3VF3', 'Cell6UMC', 'Cell7UMC', 'CMM', 'CLES', 'VF*', 'CMM_PRO'],
    'Threads': ['HPlus*'], 'Inserts': ['INSRT', 'PFIT'], 'Critical Deburr': ['DEBURCR'],
    'Cleaning': ['CLNAPXA', 'CLEAN', 'CLEANDI', 'CLEAN13', 'CLEANVC', 'CLEANWI', 'CLEANEFP', 'CLP', 'CLEP',
                 'CLQAF', 'CLEAN_IP', 'CLES', 'CLEST1', 'CNPT1', 'MetalCln', 'CLEANSP', 'ASMB*'],
    'SubCon': ['SubCA', 'SUBOP'], 'Assembly': ['ASMBL', 'ASSY'], 'Hardware': ['INSTL'],
    'Masking': ['MASK'], 'Labelling': ['APPLY'], 'Cosmetic': ['SAND', 'BLKLIN'], 'Part_Mark': ['MARK'],
    'Leak': ['LEAK'], 'Thermoform': ['DRAPE', 'STRPF', 'FORM', 'FORM4', 'ROUT'],
    'Metal': ['Metal', 'Meta_VX', 'Metal_FX', 'MetalQAF'], 'Debris': ['FLOW'], 'Annealing': ['ANNEA'],
    'Bond & Weld': ['Weld', 'SBOND', 'BNDS', 'FUWLD']
    }

tempCat = pd.DataFrame(columns=['Operations'], index=opsDict.keys())
tempCat['Operations'] = opsDict.values()
tempCat.reset_index(inplace=True)
tempCat.columns = ['Category', 'Operations']
categoryDF = tempCat

# Identify Defect Categories for given Part#
# Select data for 4 months, and use that to develop defect Categories
# vals = [3,4, 6, 7,9,10, 11]
vals = [4, 7, 9, 11]
# prod = productionDF[productionDF.Month.isin(vals) == True]
prod = productionDF
t = Timer()
t.start()

# opsDF['Operations'] = categoryDF['Operations'].apply(lambda x:  defectOps(x))
# For given Part#, select defect categories
opsDF['Operations'] = opsDF['PartNum'].apply(lambda x: getOps(x, prod))

print("{:>25} {:,} rows; {:>20}".format("Loading Operations:", productionDF.shape[0], "\tLambda Expression"), end="\t")
t.stop()

opsDF['Category'] = opsDF['Operations'].apply(lambda x: defectCategory(x, opsDict))

# Create the final dataframe for use in data analysis
ordersDF = makeFinalDF(opsDF, grpByMonth, productionDF)
# print("\t\t\nOrders", ordersDF.head())

prodDF = pd.read_csv("data/production/Production.csv")

depts = ['CNC', 'Lathe', 'Weld', 'Metal', 'Fab', 'Thermoform', 'Assembly',
         'CleanRoomAssy', 'CleanRoom', 'Bonding', 'SubCon', 'Shipping', 'QA']

clean_ops = pd.read_csv("data/production/cleaning code.csv")

# Dictionary of keywords identfying Department
deptDict = {
    'CNC': ['Cell4_GR', 'Cell1VF3', 'Cell1VF6', 'Cell3VF3', 'Cell6UMC', 'Cell7UMC', 'CMM', 'CLES', 'ANNEA', 'Cell1VF3',
            'Cell1VF6', 'Cell3VF3', 'Cell6UMC'],
    'Lathe': ['HPlus', 'HPlus', 'HPlus', 'HPlus', 'HPlus', 'HPlus', 'HPlus', 'HPlus', 'HPlus', 'HPlus', 'HPlus',
              'HPlus'],
    'Fabrication': ['INSRT', 'PFIT', 'DEBUR', 'DEBURCR', 'INSTL', 'MARK', 'DEBUR', 'DEBURCR', 'INSTL', 'MARK'],
    'CleanRoom': list(clean_ops['Operation']),
    'SubCon': ['SubCA', 'SUBOP', 'SubCA', 'SUBOP', 'SubCA', 'SUBOP', 'SUBOP', 'SubCA', 'SUBOP', 'SubCA', 'SUBOP'],
    'Assembly': ['ASMBL', 'ASSY', 'ASMBL', 'ASSY', 'ASMBL', 'ASSY', 'ASSY', 'ASMBL', 'ASSY', 'ASMBL', 'ASSY'],
    'QA': ['QAFAI', 'QAIPI', 'QAFNAL', 'MetalQAF', 'QAFAI', 'QAIPI', 'QAFNAL', 'MetalQAF'],
    'Thermoform': ['DRAPE', 'STRPF', 'FORM', 'FORM4', 'ROUT', 'FORM', 'FORM4', 'ROUT'],
    'Metal': ['VF', 'Metal', 'Meta_VX', 'Metal_FX', 'Metal_LP', 'Meta_VX', 'Metal_FX', 'Metal_LP'],
    'Bond': ['SBOND', 'BNDS', 'SBOND', 'BNDS', 'SBOND', 'BNDS', 'SBOND', 'BNDS', 'SBOND', 'BNDS'],
    'Weld': ['Weld', 'FUWLD', 'Weld', 'FUWLD', 'Weld', 'FUWLD', 'Weld', 'FUWLD', 'Weld', 'FUWLD']
    }

deptDF = pd.DataFrame(columns=['Dept'], index=deptDict.keys()).reset_index()
deptDF.drop('Dept', axis=1, inplace=True)
deptDF.columns = ['Dept']

for i in range(10):
    try:
        deptDF['OpID' + str(i + 1)] = deptDF['Dept'].apply(lambda x: deptDict[x][i])
    except IndexError:
        i += 1
        continue

mergedDF = prodDF[['Date', 'JobNum', 'PartNum', 'OpID', 'Qty']].merge(
    deptDF[['Dept', 'OpID1', 'OpID2', 'OpID3', 'OpID4', 'OpID5', 'OpID6', 'OpID7', 'OpID8']],
    left_on='OpID', right_on='OpID1')

cols = ['Date', 'Job', 'PartNum', 'Dept', 'QtyIn', 'QtyOut', 'NumMins', 'FPY(%)']

yieldDF = pd.DataFrame(columns=cols)
tset, k = set(), 0
# print("{:>10} \t {:>10} \t {:>10} \t {:>10} \t\t {:>10} \t {:>12}".\
#               format('Job', 'Dept', 'Max', 'Min', 'NumMins', 'FPY'))
t = Timer()
t.start()

for j in mergedDF['JobNum'].unique():
    # if j in list(tset): continue
    try:
        getFPY(j, mergedDF, yieldDF)
        # print(k, end=", ")
    except ValueError:
        continue
    k += 1
    # if k>=50: break
print("{:>25} {:,} rows; {:>20} ".format("Merging Process Yeild:", mergedDF.shape[0], "\tNested For-LOOP"),  end="\t")
t.stop()

deptFPY = yieldDF[['Job', 'Dept', 'FPY(%)']].groupby('Dept').mean('FPY(%)').reset_index()

fpyDF = yieldDF
# print(deptFPY)
