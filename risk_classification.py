#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import re
import matplotlib.pyplot as plt

# Import self created modules
from timing import Timer
t = Timer()
t.start()
# Import the dataframes
from DMR import dmrDF
from production import ordersDF, fpyDF, categoryDF, deptDict
from classificationFunctions import *

print("\tImport Self-created modules \t", t.stop())

orderDF = ordersDF
dmDF = dmrDF

# Analysis by Category
t.start()
# Which Categories had the highest failed quantities?

# Add a column for each Category with respective quantities for each month
# colums = ['Date', 'Qty', 'Surface Finish', 'Dimension', 'Threads', 'Cleaning',  'Scratches', 'Subcontractor',
#           'GD&T', 'Transit Damaged']
# colums = ['GD&T', 'Threads', 'Inserts', 'Critical Deburr', 'Cleaning', 'SubCon', 'Assembly', 'Hardware',
#           'Masking', 'Labelling', 'Cosmetic', 'Part_Mark', 'Leak', 'Thermoform', 'Metal', 'Debris',
#           'Annealing', 'Bond & Weld', 'Qty']
colums = ['Part_Num', 'Defect_Type', 'Defect_lem', 'Category', 'Qty', 'Surface Finish', 'Part Mark', 'Cracked',
          'Subcontractor', 'Incorrect Part', 'Cosmetic', 'Paint', 'Transit Damaged', 'Silk Screen','Burrs', 'Assembly',
          'Dimension', 'GD&T', 'Missing Fixture', 'Leaking','Material', 'Color', 'Scratches', 'Threads', 'Cleaning']
dmDF = categoryColumns(dmDF, 'Category')
# print(dmDF.columns)
dmrGrpQty = grpByQty(dmDF, colums)

# Add a column for each Category with respective quantities fro each month
colums = orderDF.columns

orderDF = categoryColumns(orderDF, 'Category')

# Determine yield by Category
fpyDF['Month'] = pd.DatetimeIndex(fpyDF['Date']).month

colums = fpyDF['Dept'].unique()
fpyDF = categoryColumns(fpyDF, 'Dept', colums)

fpyGrpYld = fpyDF[['SubCon', 'Bond', 'Fabrication', 'CNC', 'Assembly', 'Thermoform','CleanRoom', 'Month', 'FPY(%)']].groupby('Month').mean('FPY(%)')
fpyGrpYld = fpyGrpYld.reset_index()

# Create Date Column for each dataframe
dmrGrpQty['Date'] = makeDates(dmrGrpQty['Date'])
dmrGrpQty['Date'] = pd.to_datetime(dmrGrpQty['Date'], errors='coerce')

fpyGrpYld['Date'] = makeDates(fpyGrpYld['Month'])
fpyGrpYld['Date'] = pd.to_datetime(fpyGrpYld['Date'], errors='coerce')

orderDF['Date'] = makeDates(orderDF['Month'])
orderDF['Date'] = pd.to_datetime(orderDF['Date'], errors='coerce')

# Determine the annual trend of each Category
# from charts import plotLine

cols = ['Date', 'Month_name','Qty']
# plotLine(dmrGrpQty, 'Month_name', cols)

# Determine annual slope, its significance and error rates

import datetime as dt
from scipy import stats

# Determine annual slope for each data dataframe

# Annual slope for DMR
cols = ['Date', 'Month_name','Qty']
dmrSlopes = annualSlope(dmrGrpQty, cols) #.drop('date_ordinal', axis=0)

cols = ['Date','Month', 'FPY(%)']
fpySlopes = annualSlope(fpyGrpYld, cols)
fpySlopes.sort_values(by='p_value')

# Annual slope for Yield
means = fpyDF[['Dept', 'FPY(%)']].groupby('Dept').mean('FPY(%)').reset_index()
fpySlopes = fpySlopes.merge(means, left_on='Category', right_on='Dept')
fpySlopes = fpySlopes.drop('Dept', axis=1)
fpySlopes.columns = ['Dept', 'Slope', 'r_value', 'std_err', 'p_value', 'FPY(%)']

# Annual slope for Orders
pd.set_option('display.max_colwidth', 600)
cols = ['Date','PartNum', 'Description','Operations','Category', 'Month','Qty']
orderSlopes = annualSlope(orderDF, cols)

print('\nInitial Shape', orderSlopes.shape, end=" ")
orderSlopes['Dept'] = ['','Fabrication','CNC', '', 'Bond', 'SubCon','Fabrication', 'Assembly', 'CleanRoom','Fabrication', 
                       'Thermoform','Metal','SubCon','','CNC','','Fabrication' ,'Assembly']  
                      
orderSlopes = orderSlopes.merge(fpySlopes[['Dept', 'FPY(%)']], on='Dept').reset_index()
orderSlopes = orderSlopes.reindex(columns=['Dept',' Category', 'Slope', 'r_value', 'std_err', 'p_value', 'FPY(%)'])

# Get the measure of CDR of each Category and update each Category in Slopes
cdr = pd.read_csv("data/production/CDR.csv")
dmrSlopes = dmrSlopes.merge(cdr[['Category', 'cdr']], on='Category')

ordSlopes = orderSlopes.merge(cdr[['Category', 'cdr']], left_on='Dept', right_on='Category')

dmSlopes = riskProfile(dmrSlopes)
ordSlopes = riskProfile(ordSlopes)

# Modify Risk profile based on levels of Criticality, APN and Risk
dmSlopes[['Criticality', 'APN', 'RPN', 'Risk']] = digRisk(dmSlopes[['Criticality', 'APN', 'RPN', 'Risk']])

# Modify Risk profile based on levels of Criticality, APN and Risk
ordSlopes[['Criticality', 'APN', 'RPN', 'Risk']] = digRisk(ordSlopes[['Criticality', 'APN', 'RPN', 'Risk']])
print('\nFinal Shape', ordSlopes.shape)
# Create column for each risk Category, label encode 1 
val = 1
cat = 'Dimension'
for cat in dmSlopes['Category'].unique():
    try:
        # dmCats = dmDF.query(cat + ' == @val')
        # dmCats[cat]  = list(dmSlopes['Risk'][dmSlopes['Category']==cat])[0]
        dmDF[cat] = dmDF[cat].apply(lambda x: list(dmSlopes['Risk'][dmSlopes['Category']==cat])[0] if x==1 else 'No Risk')
        # dmDF[cat] = dmDF[cat].apply(lambda x: list(dmSlopes['Risk'][dmSlopes['Category']==cat])[0] if x==1 else 0)
        # print('Done: ', dmCats[cat])
    except NameError: 
        print('\t\tNameError: ', cat)
        continue
    except:
        # print('\t\tUndefinedVariableError: ', cat)
        continue
print('Time after Imports, ',end=" ")
t.stop()

def getSome(df):
    temp3 = df.iloc[:,6:]
    temp3.insert(loc=0, column='Category', value=df['Category'])
    return temp3


print("\n\t\tDMR Annual slopes\n",
    getSome(dmrSlopes))

print("\n\t\tRisk Profile\n",
    getSome(dmSlopes))

print("\n\t\tOrder Risk Profile\n",
    ordSlopes)

# print("\nInitial Order Categories\n", orderSlopes['Category'].unqiue())
print("\nFinal Order Categories\n", ordSlopes['Category'].unqiue())