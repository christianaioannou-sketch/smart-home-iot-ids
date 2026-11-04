"""
on_time_script.py  (cleaned)
----------------------------
Temporal onset analysis. Reads the per-interval PREDICTION CSVs produced by
ids_fisher.py (each has an "Original" column = ground-truth label and one
prediction column per model) and reports, per model, how close to attack
onset each interval is detected. Writes ON_TIME.txt.

HOW TO RUN
  - Put the prediction CSVs in PREDICTIONS_DIR.
  - Set NUM_FEATURES to the feature count those predictions correspond to.
  - python on_time_script.py

NOTE
  - Original hard-coded Windows paths and prompted for the feature count via
    input(); both are now in the CONFIG block. In-body duplicate imports were
    left untouched to preserve behaviour.
"""

import os

# =========================== CONFIG (edit me) ============================
PREDICTIONS_DIR = "./predictions"   # folder with the per-interval prediction CSVs
OUTPUT_DIR = "./predictions"        # where ON_TIME.txt is written
NUM_FEATURES = 30                   # feature count these predictions correspond to
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ========================================================================



import os
import pandas as pd
import csv
import re
import time

from pandas import DataFrame
import threading
import multiprocessing
import logging
import math
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns

from sklearn.model_selection import train_test_split,cross_val_score


import numpy as np
import matplotlib.pyplot as plt

num_features = NUM_FEATURES

path_app = os.path.join(PREDICTIONS_DIR, "")
path_to_readme = os.path.join(OUTPUT_DIR, "")
f_readme = open(path_to_readme + "ON_TIME.txt", "a")



count_1=0
count_BLR=0
count_KNN=0
count_SVM=0
count_DT=0
count_RF=0
count_XG=0
count=0
count_files=0

time_BLR=0
time_KNN=0
time_SVM=0
time_DT=0
time_RF=0
time_XG=0

files = (path_app)
def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

data = []
DoS = []
for data_file in sorted(os.listdir(files), key=numericalSort):
    with open(path_app + data_file) as f:
        a = sum(1 for line in f)
    row_count = sum(1 for row in data_file)  # fileObject is your csv.reader
    DoS.append(a - 1)
    data.append(data_file)
eval_BLR=0
eval_KNN=0
eval_SVM=0
eval_DT=0
eval_RF=0
eval_XG=0

for i in range(0, len(DoS)):
    count= count+1
    count_1=0
    flag_LG = False
    flag_KNN = False
    flag_SVM = False
    flag_DT = False
    flag_RF = False
    flag_XG = False
    count_similar_BLR=0
    count_similar_KNN=0
    count_similar_SVM=0
    print(data[i])
    df1 = pd.read_csv(path_app + data[i])  # read every file
    count_files=count_files+1

    for i in range (0,len(df1)):
        if df1['Original'][i]==1:

            count_1=count_1+1
            count=i
            if df1['Prediction_BLR'][i]==df1['Original'][i] and count_1==1:
                flag_LG=True
                count_BLR=count_BLR+1
                f_readme.write("\nBLR ON TIME in eval"+str(count_files))
                print("on time")


            if df1['Prediction_BLR'][i]==df1['Original'][i] and count_1>=1 and flag_LG==False:
                print(count_1,flag_LG)

                if count_1>time_BLR:
                    time_BLR=count_1
                    eval_BLR=count_files
                    flag_LG=True
                    index, count_MPI_BLR = np.unique(df1['Original'].values, return_counts=True)

                else:
                    flag_XG=True
            if df1['Prediction_KNN'][i] == df1['Original'][i] and count_1==1:
                f_readme.write("\nKNN ON TIME in eval" + str(count_files))
                count_KNN = count_KNN + 1
                flag_KNN = True
                print("on time")

            if df1['Prediction_KNN'][i]==df1['Original'][i] and count_1>=1 and flag_KNN==False:
                print(count_1,time_KNN,count_files)
                if count_1>time_KNN:
                    time_KNN=count_1
                    eval_KNN=count_files
                    flag_KNN=True
                    index, count_MPI_KNN = np.unique(df1['Original'].values, return_counts=True)
                else:
                    flag_KNN=True


            if df1['Prediction_SVM'][i] == df1['Original'][i] and count_1==1:
                f_readme.write("\nSVM ON TIME in eval" + str(count_files))
                flag_SVM = True
                count_SVM = count_SVM + 1
                print("on time")
            if df1['Prediction_SVM'][i]==df1['Original'][i] and count_1>=1 and flag_SVM==False:
                if count_1>time_SVM:
                    time_SVM=count_1
                    eval_SVM=count_files
                    flag_SVM=True
                    index, count_MPI_SVM = np.unique(df1['Original'].values, return_counts=True)
                else:
                    flag_SVM=True


            if df1['Prediction_DT'][i] == df1['Original'][i] and count_1==1:
                f_readme.write("\nDT ON TIME in eval" + str(count_files))
                flag_DT = True
                count_DT = count_DT + 1
                print("on time")
            if df1['Prediction_DT'][i]==df1['Original'][i] and count_1>1 and flag_DT==False:
                if count_1>time_DT:
                    time_DT=count_1
                    eval_DT=count_files
                    flag_DT = True
                    index, count_MPI_DT= np.unique(df1['Original'].values, return_counts=True)
                else:
                    flag_DT=True
            if df1['Prediction_RF'][i] == df1['Original'][i] and count_1==1:
                f_readme.write("\nRF ON TIME in eval" + str(count_files))
                count_RF = count_RF + 1
                flag_RF = True
                print("on time")

            if df1['Prediction_RF'][i] == df1['Original'][i] and count_1 > 1 and flag_RF == False:
                if count_1 >= time_RF:
                    time_RF = count_1
                    eval_RF = count_files
                    flag_RF=True
                    index, count_MPI_RF= np.unique(df1['Original'].values, return_counts=True)
                else:
                    flag_RF=True

            if df1['Prediction_XG'][i] == df1['Original'][i] and count_1==1:
                f_readme.write("\nXG ON TIME in eval" + str(count_files))
                flag_XG = True
                count_XG = count_XG + 1
                print("on time")

            if df1['Prediction_XG'][i] == df1['Original'][i] and count_1 > 1 and flag_XG == False:
                if count_1 >= time_XG:
                    time_XG = count_1
                    eval_XG = count_files
                    flag_XG=True
                    index, count_MPI_XG = np.unique(df1['Original'].values, return_counts=True)
                else:
                    flag_XG=True

print("TIME BLR",time_BLR,"on file:",eval_BLR)
print("TIME KNN",time_KNN,"on file:",eval_KNN)
print("TIME SVM",time_SVM,"on file:",eval_SVM)
print("TIME DT",time_DT,"on file:",eval_DT)
print("TIME RF",time_RF,"on file:",eval_RF)
print("TIME XG",time_XG,"on file:",eval_XG)

f_readme.write("\n\nTotal BLR ON TIME: "+str(count_BLR))
f_readme.write("\nTotal KNN ON TIME: "+str(count_KNN))
f_readme.write("\nTotal SVM ON TIME: "+str(count_SVM))
f_readme.write("\nTotal DT ON TIME: "+str(count_DT))
f_readme.write("\nTotal RF ON TIME: "+str(count_RF))
f_readme.write("\nTotal XGBOOST ON TIME:"+str(count_XG))

if time_BLR>0:
    f_readme.write("\n\nBLR found the attack after "+str(time_BLR-1)+" MPI on a total "+str(count_MPI_BLR[1]))
else:
    f_readme.write("\n\nBLR found the attack after " + str(time_BLR) + " MPI" )

if time_KNN>0:
    f_readme.write("\nKNN found the attack after "+str(time_KNN-1)+" MPI on a total "+str(count_MPI_KNN[1]))
else:
    f_readme.write("\nKNN found the attack after " + str(time_KNN) + " MPI " )
if time_SVM>0:
    f_readme.write("\nSVM found the attack after "+str(time_SVM-1)+" MPI on a total "+str(count_MPI_SVM[1]))
else:
    f_readme.write("\nSVM found the attack after " + str(time_SVM) + " MPI ")

if time_DT>0:
    f_readme.write("\nDT found the attack after "+str(time_DT-1)+" MPI on a total "+str(count_MPI_DT[1]))
else:
    f_readme.write("\nDT found the attack after " + str(time_DT) + " MPI" )

if time_RF>0:
    f_readme.write("\nRF found the attack after "+str(time_RF-1)+" MPI on a total "+str(count_MPI_RF[1]))
else:
    f_readme.write("\nRF found the attack after " + str(time_RF) + " MPI " )

if time_XG>0:
    f_readme.write("\nXGBOOST found the attack after "+str(time_XG-1)+" MPI on a total "+str(count_MPI_XG[1]))
else:
    f_readme.write("\nXGBOOST found the attack after " + str(time_XG) + " MPI" )

f_readme.close()