import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

# import import_ipynb

# from ProductionData import productionDF
from timing import Timer
# from ProductionData import getOp, defectCategory, opsDict

# from Interviews import getLemma, removeStops


def getData(path, files, cols=None):
    box = []
    for f in files:
        doc = pd.read_csv(path + f + ".csv", usecols=cols)
        box.append(doc)
    if len(box) > 1:
        df = box[0].append(box[1:], ignore_index=True).dropna()
    else:
        df = box[0]
    return df


def removeStops(arr):
    print(arr)
    arr = [re.sub(r'(\d*|Chin|Chiltz)', '', arr)]
    print(arr)
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(arr)
    return vectorizer.get_feature_names_out()


def categorize(d):
    mat = []
    if d.isdigit():
        mat.append("Numerical")
    if len(re.findall(r"(?:cracked|crack*\w*)", d.lower())) > 0:
        mat.append('Cracked')
    if len(re.findall(r"(?:damage*|broke*|dent*\w*)", d.lower())) > 0:
        mat.append('Transit Damaged')
    if len(re.findall(r"(?:surface|finish*\w*)", d.lower())) > 0:
        mat.append('Surface Finish')
    if len(re.findall(r"(?:dim|dimension\w*)", d.lower())) > 0:
        mat.append('Dimension')
    if len(re.findall(r"(?:perpend*|flatn*|position|tolerance*|roughness|thickness|out of|oot|oos\w*)", d.lower())) > 0:
        mat.append('GD&T')
    if len(re.findall(r"(?:thread*\w*)", d.lower())) > 0:
        mat.append('Threads')
    if len(re.findall(r"(?:leak*\w*)", d.lower())) > 0:
        mat.append('Leaking')
    if len(re.findall(r"(?:incorrect*\w*)", d.lower())) > 0:
        mat.append('Incorrect Part')
    if len(re.findall(r"(?:mark*|ink\w*)", d.lower())) > 0:
        mat.append('Part Mark')
    if len(re.findall(r"(?:color*|colour*|discolor*\w*)", d.lower())) > 0:
        mat.append('Color')
    if len(re.findall(r"(?:clean*\w*)", d.lower())) > 0:
        mat.append("Cleaning")
    if len(re.findall(r"(?:scratch*\w*)", d.lower())) > 0:
        mat.append("Scratches")
    if len(re.findall(r"(?:rotat*|assembl*\w*)", d.lower())) > 0:
        mat.append("Assembly")
    if len(re.findall(r"(?:paint*\w*)", d.lower())) > 0:
        mat.append("Paint")
    if len(re.findall(r"(?:silk*\w*)", d.lower())) > 0:
        mat.append("Silk Screen")
    if len(re.findall(r"(?:burr*|deburr*\w*)", d.lower())) > 0:
        mat.append("Burrs")
    if len(re.findall(r"(?:mtl*|material*\w*)", d.lower())) > 0:
        mat.append("Material")
    if len(re.findall(r"(?:subco*|vendor*|osp\w*)", d.lower())) > 0:
        mat.append("Subcontractor")
    if len(re.findall(r'(?:cosmet*\w*)', d.lower())) > 0:
        mat.append("Cosmetic")
    if len(re.findall(r"(?:missing\w*)", d)) > 0:
        mat.append("Missing Fixture")
    if len(mat) == 0:
        return "Not Known"
    else:
        return mat


# from production import productionDF
def loadRMA(path):
    # Get RMA data
    colums = ['RMA Number', 'Part Number', 'Return  Qty', 'RMA Date', 'Customer', 'Responsibility', 'Type of Issue',
              'Notes']

    rma2021DF = getData(path, ['RMA 2021'], colums)

    from Interviews import removeStops, getLemma

    # Drop stop words from each row in Defect column
    rma2021DF['Defect_lem'] = rma2021DF['Notes'].apply(lambda x: removeStops([x]))

    defects = ['scratched', 'surface', 'damage', 'color', 'mark', 'cleaning', 'scratch']
    # ncr_DF['defect_vect'] = ncr_DF['Defect'].apply(lambda x: removeStops([str(x)]))

    # Identify Category from key words in each defect
    rma2021DF['Category'] = rma2021DF['Defect_lem'].apply(lambda x: categorize(', '.join(x)))

    # Create final dataframe by selecting columns and re-ordering them
    temp = ''
    temp = rma2021DF[['RMA Number', 'Part Number', 'Return  Qty', 'RMA Date',
                      'Type of Issue', 'Defect_lem', 'Category']]
    temp.columns = ['RMA#', 'Part_Num', 'Qty', 'Date', 'Defect_type', 'Defect_lem', 'Category']
    temp['RMA#'] = temp['RMA#'].apply(lambda x: 'RMA ' + str(x))
    temp = temp.reindex(columns=['Date', 'RMA#', 'Part_Num', 'Defect_type', 'Defect_lem', 'Category', 'Qty'])
    temp.columns = ['Date', 'Doc_num', 'Part_Num', 'Defect_Type', 'Defect_lem', 'Category', 'Qty']
    rmaDF = temp
    return temp


def loadNCR(path):
    # Prepare NCR data, append data from all files
    colums = ['Issued Date', 'NCR #', 'Job Number', 'Part Number', 'Rev', 'NCR Qty', 'Description of Defect',
              'Defect Type']
    files = ['Jan NC 2021', 'Feb NC 2021', 'Mar NC 2021', 'April NC 2021', 'May NC 2021',
             'June NC 2021', 'July NC 2021', 'Aug-Oct NC 2021', 'Nov NC 2021', 'Dec NC 2021']

    ncr_DF = getData(path, files, colums)

    # Rename the columns
    ncr_DF.columns = ['Date', 'NCR_num', 'Job_Number', 'Part_Number', 'Rev', 'Qty',
                      'Defect', 'Defect_Type']

    ncr_DF['Defect_lem'] = ncr_DF['Defect'].apply(lambda x: (getLemma(x)))

    # Categorize each defect
    ncr_DF['Category'] = ncr_DF['Defect_lem'].apply(lambda x: categorize(str(x)))
    ncr_DF['NCR_num'] = ncr_DF['NCR_num'].apply(lambda x: 'NCR ' + str(x).strip('.0'))

    # Create final dataframe
    temp1 = ''
    temp1 = ncr_DF[['Date', 'NCR_num', 'Part_Number', 'Qty', 'Defect_lem', 'Defect_Type', 'Category']]

    temp1 = temp1.reindex(columns=['Date', 'NCR_num', 'Part_Number', 'Defect_type', 'Defect_lem', 'Category', 'Qty'])
    cols = ['Date', 'Doc_num', 'Part_Num', 'Defect_Type', 'Defect_lem', 'Category', 'Qty']
    temp1.columns = cols
    return temp1


def loadQN(path):
    amat = getData(path, ['AMAT QN - Apr2022'])
    amat['Date Opened'] = pd.to_datetime(amat['Date Opened'])
    amat['Qty'] = np.random.randint(1, 6, size=(len(amat), 1))  # Assign random Qty for each row

    # Select columns required and rename them
    temp1 = amat[['Date Opened', 'QN#', 'Part number', 'Title', 'Defect Type', 'Qty']]

    temp1.columns = ['Date', 'Doc_num', 'PartNum', 'Defect', 'Defect_type', 'Qty']

    productionDF['partNoRev'] = productionDF['PartNum'].apply(lambda x: str(x).split(',')[0])

    # Select data for 4 months, and use that to develop defect Categories
    vals = [3, 4, 6, 7, 9, 10, 11]
    # vals = [4, 7,9,11]
    prod = productionDF[productionDF.Month.isin(vals) == True]
    t = Timer()
    t.start()

    # opsDF['Operations'] = categoryDF['Operations'].apply(lambda x:  defectOps(x))
    temp1['Operations'] = temp1['PartNum'].apply(lambda x: getOp(x, prod))
    t.stop()

    #  Breakdown the Defect column into keywords
    temp1['Category'] = temp1['Operations'].apply(lambda x: defectCategory(x, opsDict))
    temp1 = temp1[temp1['Category'].apply(lambda x: len(x) > 0)]
    temp1.drop('Operations', axis=1, inplace=True)

    # from Interviews import removeStops, getLemma

    # Remove stop words from Defect column
    temp1['Defect_lem'] = temp1['Defect'].apply(lambda x: removeStops([x]))

    qnDF = temp1.reindex(columns=['Date', 'Doc_num', 'Part_Num', 'Defect_Type', 'Defect_lem', 'Category', 'Qty'])

    qnDF.columns = ['Date', 'Doc_num', 'Part_Num', 'Defect_Type', 'Defect_lem', 'Category', 'Qty']

    return qnDF


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = "data/FY2021/"
    rmaDF = loadRMA(path)
    ncrDF = loadNCR(path)
    qnDF = loadQN(path)

    # Combine RMA,QN and NRC dataframes
    tempDF = ''
    tempDF = ncrDF.append([rmaDF, qnDF], ignore_index=True)
    tempDF['Date'] = pd.to_datetime(tempDF['Date'], errors='coerce').dropna(inplace=False)
    dmrDF = temp
    print(dmrDF.head())


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
