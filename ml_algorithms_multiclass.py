"""
ml_algorithms_multiclass.py  (cleaned)
--------------------------------------
Multiclass variant of the experiment runner: predicts the attack TYPE
(not just benign/malicious). Reads the ORIGINAL multiclass training/evaluation
CSVs (label column "Class"), scales, then trains and evaluates the six models
and writes confusion matrices, metrics and plots to OUTPUT_DIR.

HOW TO RUN
  - Set TRAIN_CSV / EVAL_CSV / OUTPUT_DIR below, then:
    python ml_algorithms_multiclass.py

NOTE
  - Hard-coded Windows paths replaced by the CONFIG block. Label column must be
    "Class". In-body duplicate imports left as-is to preserve behaviour.
"""

import os

# =========================== CONFIG (edit me) ============================
TRAIN_CSV = "training_1sec_ORIGINAL.csv"
EVAL_CSV  = "evaluation_1sec_ORIGINAL.csv"
OUTPUT_DIR = "./results_multiclass"
LABEL_COL = "Class"
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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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
import seaborn as sns
import matplotlib.pyplot as plt

df1 = pd.read_csv(TRAIN_CSV)
df2 = pd.read_csv(EVAL_CSV)
path_to_file = os.path.join(OUTPUT_DIR, "")


from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, classification_report
f = open(path_to_file+"ML_results_python.txt", "a")




X_train=df1.iloc[:,1:len(df1.columns)]
X_test=df2.iloc[:,1:len(df2.columns)]
y_train=df1["Class"]
y_test=df2["Class"]
print(X_train)

###################################### after spliting perform scaling ########################
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(X_train)
xtest = sc_x.transform(X_test) ##### use transform because we want the same scaler as in the training set



#pd.DataFrame(xtest).to_csv("../test.csv")
#pd.DataFrame(xtrain).to_csv("../train.csv")
###################################### apply the ML algorithm ##################################
classifier=LogisticRegression(multi_class='multinomial', solver='lbfgs' ,penalty='none',C=10,max_iter=10000)
classifier.fit(xtrain,y_train)
y_final=classifier.predict(xtest)
print(confusion_matrix(y_test,y_final))
print(y_final)
f.write("---------------------------------------- LOGSTIC REGRESSION ----------------------------------------------------------\n")
f.write('\nConfusion matrix before grid search:'+str(confusion_matrix(y_test,y_final)))
print("######################## K fold cross validation results ###########################")
_scoring = ['balanced_accuracy']

# define search space
space = dict()
#space['solver'] = [ 'lbfgs', 'liblinear']
#pace['penalty'] = ['none', 'l1', 'l2']
space['C'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
LRparam_grid =[
    {'solver': ['lbfgs'],'penalty': ['l2','none'],'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]},
{'solver': ['sag'],'penalty': ['l2','none'],'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]},
{'solver': ['saga'],'penalty': ["elasticnet", "l1", "l2", "none"],'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]},

]



cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

################### GRID SEARCH RESULTS ###########################
'''score= GridSearchCV(classifier,LRparam_grid,cv=cv ,refit=True, verbose=2,scoring="balanced_accuracy")
result=score.fit(xtrain,y_train)
#scores=pd.DataFrame(score)
#print(scores)
# summarize result
best=result.best_estimator_
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

f.write('Best parameters:'+str(result.best_params_))
#print("Mean of parameters {} %".format(scores.mean()*100))
y_final=best.predict(xtest)'''
print(classification_report(y_test,y_final))
print(confusion_matrix(y_test,y_final))
accuracy_LR=metrics.balanced_accuracy_score(y_test, y_final)

#f1_score_LR=metrics.f1_score(y_test, y_final)

MCC_LR=metrics.matthews_corrcoef(y_test, y_final)
kappa_LR=metrics.cohen_kappa_score(y_test,y_final)
print("Accuracy:",metrics.accuracy_score(y_test, y_final))
print("Balanced Accuracy:",metrics.balanced_accuracy_score(y_test, y_final))
#print("Precision:",metrics.precision_score(y_test, y_final))
#print("Recall:",metrics.recall_score(y_test, y_final))
#print("F1-score:",metrics.f1_score(y_test, y_final))
print("MCC:",metrics.matthews_corrcoef(y_test, y_final))
print('Kappa score:',metrics.cohen_kappa_score(y_test,y_final))
y_pred_proba = classifier.predict_proba(xtest)[::,1]
#fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
#auc = metrics.roc_auc_score(y_test, y_pred_proba)
#print("AUC:",auc)
f1_score_LR= metrics.f1_score(y_test, y_final, average='macro')
LR_report_DT=classification_report(y_test,y_final, output_dict=True)
LR_f1_classes=[]
for i in range(0,5):
    LR_f1_classes.append(LR_report_DT[str(i)]['f1-score'])

print(metrics.f1_score(y_test, y_final, average='micro'))
cm=confusion_matrix(y_test,y_final)
f1_micro=metrics.f1_score(y_test, y_final, average='micro')

FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP)
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
print("This is FPR",FPR)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy for each class
ACC = (TP+TN)/(TP+FP+FN+TN)


f.write("\nFPR :"+str( FPR))
f.write("\nF1-micro"+str(f1_micro))
ax = sns.heatmap(cm, annot=True, cmap='Blues')

ax.set_title('Logistic Regression Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Category')
ax.set_ylabel('Actual Category ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Benign','Mirai', 'MiTM','DoS','Scanning'])
ax.yaxis.set_ticklabels(['Benign','Mirai', 'MiTM','DoS','Scanning'])
plt.savefig(path_to_file+'LG_cm.png',bbox_inches='tight')
## Display the visualization of the Confusion Matrix.
plt.show()



f.write('\nClassification report'+str(LR_report_DT))
f.write(str(confusion_matrix(y_test,y_final)))
f.write("\nAccuracy:"+ str(metrics.accuracy_score(y_test, y_final)))
f.write("\nBalanced Accuracy:"+str(metrics.balanced_accuracy_score(y_test, y_final)))
#f.write("\nPrecision:"+str(metrics.precision_score(y_test, y_final)))
#f.write("\nRecall:"+str(metrics.recall_score(y_test, y_final)))
#f.write("\nF1-score:"+str(metrics.f1_score(y_test, y_final)))
f.write("\nMCC:"+str(metrics.matthews_corrcoef(y_test, y_final)))
f.write('\nKappa score:'+str(metrics.cohen_kappa_score(y_test,y_final)))
#f.write("\nAUC:"+str(auc))





print("############################################################### KNN #############################################################################")
from sklearn.neighbors import KNeighborsClassifier
classifier_KNN=KNeighborsClassifier(n_neighbors=7, metric="minkowski", p=1)
classifier_KNN.fit(xtrain,y_train)
y_pred_KNN=classifier_KNN.predict(xtest)

print(confusion_matrix(y_test,y_pred_KNN))
print("MCC:",metrics.matthews_corrcoef(y_test, y_pred_KNN))

'''print('------------------------------------Grid Search Results KNN----------------------------------------')
hyperparameters=dict()
hyperparameters['metric'] =['minkowski','euclidean','manhattan']
hyperparameters['n_neighbors'] = list(range(5,50))
hyperparameters['p']=[1,2]

grid_search_KNN = GridSearchCV(classifier_KNN, hyperparameters, refit=True, verbose=3,cv=cv,scoring="balanced_accuracy")
result=grid_search_KNN.fit(xtrain,y_train)

best_KNN=result.best_estimator_
print('-------------------------------GRID SEARCH RESULTS KNN--------------------------------')
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)
y_pred_KNN=best_KNN.predict(xtest)
'''
cm=confusion_matrix(y_test,y_pred_KNN)




print(y_pred_KNN)
accuracy_KNN=metrics.accuracy_score(y_test, y_pred_KNN)
#f1_score_KNN=metrics.f1_score(y_test, y_pred_KNN)
MCC_KNN=metrics.matthews_corrcoef(y_test, y_pred_KNN)
print(classification_report(y_test,y_pred_KNN))
kappa_KNN=metrics.cohen_kappa_score(y_test,y_pred_KNN)
print("Accuracy:",metrics.balanced_accuracy_score(y_test, y_pred_KNN))
print("Balanced Accuracy:",metrics.balanced_accuracy_score(y_test, y_pred_KNN))
#print("Precision:",metrics.precision_score(y_test, y_pred_KNN))
#print("Recall:",metrics.recall_score(y_test, y_pred_KNN))
#print("F1-score:",metrics.f1_score(y_test, y_pred_KNN))
print("MCC:",metrics.matthews_corrcoef(y_test, y_pred_KNN))
print('Kappa score:',metrics.cohen_kappa_score(y_test,y_pred_KNN))
#y_pred_proba = classifier_KNN.predict_proba(xtest)[::,1]
#fpr2, tpr2, _ = metrics.roc_curve(y_test,  y_pred_proba)
#auc = metrics.roc_auc_score(y_test, y_pred_proba)
#print("AUC:",auc)


ax = sns.heatmap(cm, annot=True, cmap='Blues')

ax.set_title('KNN Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Category')
ax.set_ylabel('Actual Category ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Benign','Mirai', 'MiTM','DoS','Scanning'])
ax.yaxis.set_ticklabels(['Benign','Mirai', 'MiTM','DoS','Scanning'])
plt.savefig(path_to_file+'KNN_cm.png',bbox_inches='tight')
## Display the visualization of the Confusion Matrix.
plt.show()
f1_micro_KNN=metrics.f1_score(y_test, y_pred_KNN, average='micro')

FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP)
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
print("This is FPR",FPR)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy for each class
ACC = (TP+TN)/(TP+FP+FN+TN)


f.write("\nFPR :"+str( FPR))
f.write("\nF1-micro"+str(f1_micro_KNN))


f.write("\n------------------------------------ KNN ----------------------------------------------------------\n")
f.write(str(confusion_matrix(y_test,y_pred_KNN)))
f.write("\nAccuracy:"+ str(metrics.accuracy_score(y_test, y_pred_KNN)))
f.write("\nBalanced Accuracy:"+str(metrics.balanced_accuracy_score(y_test, y_pred_KNN)))
#f.write("\nPrecision:"+str(metrics.precision_score(y_test, y_pred_KNN)))
#f.write("\nRecall:"+str(metrics.recall_score(y_test, y_pred_KNN)))
#f.write("\nF1-score:"+str(metrics.f1_score(y_test, y_pred_KNN)))
f.write("\nMCC:"+str(metrics.matthews_corrcoef(y_test, y_pred_KNN)))
f.write('\nKappa score:'+str(metrics.cohen_kappa_score(y_test,y_pred_KNN)))
#f.write("\nAUC:"+str(auc))
f1_score_KNN= metrics.f1_score(y_test, y_pred_KNN, average='macro')


KNN_report_DT=classification_report(y_test,y_pred_KNN, output_dict=True)
KNN_f1_classes=[]
for i in range(0,5):
    KNN_f1_classes.append(KNN_report_DT[str(i)]['f1-score'])
#cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
#score= cross_validate(classifier_KNN,xtrain,y_train,cv=cv , scoring=_scoring,return_train_score=True)
#scores=pd.DataFrame(score)
#print(scores)
#print("Mean of parameters {} %".format(scores.mean()*100))
################################## SVM ##############################################################
from sklearn.svm import SVC

classifier_SVM= SVC(kernel="poly",C=0.75,gamma=0.8)
classifier_SVM.fit(xtrain,y_train)
print("############################################################ SVM ###################################################################")
y_pred_SVM=classifier_SVM.predict(xtest)
print(confusion_matrix(y_test,y_pred_SVM))
cm=confusion_matrix(y_test,y_pred_SVM)
print("MCC:",metrics.matthews_corrcoef(y_test, y_pred_SVM))


accuracy_SVM=metrics.balanced_accuracy_score(y_test, y_pred_SVM)


'''parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
              {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf','poly'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

grid = GridSearchCV(classifier_SVM, parameters, verbose=3, cv=cv, scoring="balanced_accuracy")
result=grid.fit(xtrain,y_train)

best_KSVM=result.best_estimator_
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

y_pred_SVM=best_KSVM.predict(xtest)'''
print(confusion_matrix(y_test,y_pred_SVM))

#f1_score_SVM=metrics.f1_score(y_test, y_pred_SVM)
MCC_SVM=metrics.matthews_corrcoef(y_test, y_pred_SVM)
print(classification_report(y_test,y_pred_SVM))
kappa_SVM=metrics.cohen_kappa_score(y_test,y_pred_SVM)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_SVM))
print("Balanced Accuracy:",metrics.balanced_accuracy_score(y_test, y_pred_SVM))
#print("Precision:",metrics.precision_score(y_test, y_pred_SVM))
#print("Recall:",metrics.recall_score(y_test, y_pred_SVM))
#print("F1-score:",metrics.f1_score(y_test, y_pred_SVM))
print("MCC:",metrics.matthews_corrcoef(y_test, y_pred_SVM))
print('Kappa score:',metrics.cohen_kappa_score(y_test,y_pred_SVM))
#fpr3, tpr3, _ = metrics.roc_curve(y_test,  y_pred_proba)
#auc = metrics.roc_auc_score(y_test, y_pred_proba)
#print("AUC:",auc)
f1_score_SVM= metrics.f1_score(y_test, y_pred_SVM, average='macro')
SVM_report_DT=classification_report(y_test,y_pred_SVM, output_dict=True)
SVM_f1_classes=[]
for i in range(0,5):
    SVM_f1_classes.append(SVM_report_DT[str(i)]['f1-score'])




ax = sns.heatmap(cm, annot=True, cmap='Blues')

ax.set_title('SVM Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Category')
ax.set_ylabel('Actual Category ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Benign','Mirai', 'MiTM','DoS','Scanning'])
ax.yaxis.set_ticklabels(['Benign','Mirai', 'MiTM','DoS','Scanning'])
plt.savefig(path_to_file+'SVM_cm.png',bbox_inches='tight')
## Display the visualization of the Confusion Matrix.
plt.show()
f1_micro_SVM=metrics.f1_score(y_test, y_pred_SVM, average='micro')

FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP)
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
print("This is FPR",FPR)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy for each class
ACC = (TP+TN)/(TP+FP+FN+TN)


f.write("\nFPR :"+str( FPR))
f.write("\nF1-micro"+str(f1_micro_SVM))



f.write("\n------------------------------------ SVM ----------------------------------------------------------\n")
f.write(str(confusion_matrix(y_test,y_pred_SVM)))
f.write("\nAccuracy:"+ str(metrics.accuracy_score(y_test, y_pred_SVM)))
f.write("\nBalanced Accuracy:"+str(metrics.balanced_accuracy_score(y_test, y_pred_SVM)))
#f.write("\nPrecision:"+str(metrics.precision_score(y_test, y_pred_SVM)))
#f.write("\nRecall:"+str(metrics.recall_score(y_test, y_pred_SVM)))
#f.write("\nF1-score:"+str(metrics.f1_score(y_test, y_pred_SVM)))
f.write("\nMCC:"+str(metrics.matthews_corrcoef(y_test, y_pred_SVM)))
f.write('\nKappa score:'+str(metrics.cohen_kappa_score(y_test,y_pred_SVM)))
#f.write("\nAUC:"+str(auc))





############################## KERNEL SVM ##############################################################


''''print("###################################################### KERNEL SVM #############################################################")
classifier_KERNEL_SVM= SVC(kernel="rbf")
classifier_KERNEL_SVM.fit(xtrain,y_train)
y_pred=classifier_KERNEL_SVM.predict(xtest)
print(confusion_matrix(y_test,y_pred))

accuracy_KSVM=metrics.accuracy_score(y_test, y_pred)
#f1_score_KSVM=metrics.f1_score(y_test, y_pred)
MCC_KSVM=metrics.matthews_corrcoef(y_test, y_pred)
f1_score_KSVM= metrics.f1_score(y_test, y_pred, average='macro')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Balanced Accuracy:",metrics.balanced_accuracy_score(y_test, y_pred))
#print("Precision:",metrics.precision_score(y_test, y_pred))
#print("Recall:",metrics.recall_score(y_test, y_pred))
#print("F1-score:",metrics.f1_score(y_test, y_pred))
print("MCC:",metrics.matthews_corrcoef(y_test, y_pred))
print('Kappa score:',metrics.cohen_kappa_score(y_test,y_pred))
#fpr4, tpr4, _ = metrics.roc_curve(y_test,  y_pred)
#auc = metrics.roc_auc_score(y_test, y_pred)
#print("AUC:",auc)
df_report_KSVM=classification_report(y_test,y_pred, output_dict=True)
KSVM_f1_classes=[]
for i in range(0,5):
    KSVM_f1_classes.append(df_report_KSVM[str(i)]['f1-score'])
f.write("\n------------------------------------ KERNEL ----------------------------------------------------------\n")
f.write(str(confusion_matrix(y_test,y_pred)))
f.write("\nAccuracy:"+ str(metrics.accuracy_score(y_test, y_pred)))
#f.write("\nBalanced Accuracy:"+str(metrics.balanced_accuracy_score(y_test, y_pred)))
#f.write("\nPrecision:"+str(metrics.precision_score(y_test, y_pred)))
##f.write("\nRecall:"+str(metrics.recall_score(y_test, y_pred)))
#f.write("\nF1-score:"+str(metrics.f1_score(y_test, y_pred)))
f.write("\nMCC:"+str(metrics.matthews_corrcoef(y_test, y_pred)))
f.write('\nKappa score:'+str(metrics.cohen_kappa_score(y_test,y_pred)))
#f.write("\nAUC:"+str(auc))'''


##############################  DECISION TREE  ##############################################################
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
classifier_DT = DecisionTreeClassifier(criterion = 'entropy', random_state = 0,min_samples_split=20,
                                    min_samples_leaf=5, max_depth=20)
classifier_DT.fit(xtrain, y_train)


print("############################### DECISION TREE ########################################")
y_pred=classifier_DT.predict(xtest)
print(classification_report(y_test,y_pred))
print("MCC:",metrics.matthews_corrcoef(y_test, y_pred))

'''print("-------------------------------------GRID SEARCH RESULTS DECISION TREE -------------------------------------------------")

# Create the parameter grid based on the results of random search
params = {
    'max_depth': [2, 3, 5, 10, 20, 25,30],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}
grid_search = GridSearchCV(estimator=classifier_DT,
                           param_grid=params,
                           cv=cv, verbose=3,scoring="balanced_accuracy")



result=grid_search.fit(xtrain,y_train)

best_DT=result.best_estimator_
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

y_pred=best_DT.predict(xtest)'''
cm=confusion_matrix(y_test,y_pred)
print(confusion_matrix(y_test,y_pred))

accuracy_DT=metrics.accuracy_score(y_test, y_pred)
#f1_score_DT=metrics.f1_score(y_test, y_pred)
MCC_DT=metrics.matthews_corrcoef(y_test, y_pred)
f1_score_DT= metrics.f1_score(y_test, y_pred, average='macro')
print(f1_score_DT)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Balanced Accuracy:",metrics.balanced_accuracy_score(y_test, y_pred))

print(classification_report(y_test,y_pred))
df_report_DT=classification_report(y_test,y_pred, output_dict=True)
DT_f1_classes=[]
for i in range(0,5):
    DT_f1_classes.append(df_report_DT[str(i)]['f1-score'])


ax = sns.heatmap(cm, annot=True, cmap='Blues')

ax.set_title('Decision Tree Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Category')
ax.set_ylabel('Actual Category ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Benign','Mirai', 'MiTM','DoS','Scanning'])
ax.yaxis.set_ticklabels(['Benign','Mirai', 'MiTM','DoS','Scanning'])
plt.savefig(path_to_file+'DT_cm.png',bbox_inches='tight')
## Display the visualization of the Confusion Matrix.
plt.show()

kappa_DT=metrics.cohen_kappa_score(y_test,y_pred)
#print("Precision:",metrics.precision_score(y_test, y_pred))
#print("Recall:",metrics.recall_score(y_test, y_pred))
#print("F1-score:",metrics.f1_score(y_test, y_pred))
print("MCC:",metrics.matthews_corrcoef(y_test, y_pred))
print('Kappa score:',metrics.cohen_kappa_score(y_test,y_pred))
#fpr5, tpr5, _ = metrics.roc_curve(y_test,  y_pred)
#auc = metrics.roc_auc_score(y_test, y_pred)
#print("AUC:",auc)


f1_micro_DT=metrics.f1_score(y_test, y_pred, average='micro')

FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP)
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
print("This is FPR",FPR)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy for each class
ACC = (TP+TN)/(TP+FP+FN+TN)


f.write("\nFPR :"+str( FPR))
f.write("\nF1-micro"+str(f1_micro_DT))




f.write("\n------------------------------------ DECISION TREE ----------------------------------------------------------\n")
f.write(str(confusion_matrix(y_test,y_pred)))
f.write("\nAccuracy:"+ str(metrics.accuracy_score(y_test, y_pred)))
f.write("\nBalanced Accuracy:"+str(metrics.balanced_accuracy_score(y_test, y_pred)))
#f.write("\nPrecision:"+str(metrics.precision_score(y_test, y_pred)))
#f.write("\nRecall:"+str(metrics.recall_score(y_test, y_pred)))
#f.write("\nF1-score:"+str(metrics.f1_score(y_test, y_pred)))
f.write("\nMCC:"+str(metrics.matthews_corrcoef(y_test, y_pred)))
f.write('\nKappa score:'+str(metrics.cohen_kappa_score(y_test,y_pred)))
#f.write("\nAUC:"+str(auc))

'''sort = classifier.feature_importances_.argsort()
plt.barh(df1.columns[sort], classifier.feature_importances_[sort])
plt.xlabel("Feature Importance")
plt.show()'''


############################################# RANDOM FOREST ################################################

from sklearn.ensemble import RandomForestClassifier
classifier_RF = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0,max_depth=10,min_samples_leaf=1,min_samples_split=2)
classifier_RF.fit(xtrain, y_train)
print("############################### RANDOM FOREST ########################################")
y_pred=classifier_RF.predict(xtest)
cm=confusion_matrix(y_test,y_pred)
print(confusion_matrix(y_test,y_pred))
# Create the parameter grid based on the results of random search
'''params = {

    'criterion': ["gini", "entropy"],
    'max_depth': [10, 20, 30],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2],
 'min_samples_split': [2, 5],
 'n_estimators': [200, 400, 600, 800, 1000]}

grid_search = GridSearchCV(estimator=classifier_RF,
                           param_grid=params,
                           cv=cv,  verbose=3,scoring="balanced_accuracy")



result=grid_search.fit(xtrain,y_train)

best_RF=result.best_estimator_
print('Best Score: %s' % result.best_score_)
print("Best Hyperparameters: %s" % result.best_params_)

y_pred=best_RF.predict(xtest)
'''
print(confusion_matrix(y_test,y_pred))

accuracy_RF=metrics.accuracy_score(y_test, y_pred)
f1_score_RF= metrics.f1_score(y_test, y_pred, average='macro')

MCC_RF=metrics.matthews_corrcoef(y_test, y_pred)
kappa_RF=metrics.cohen_kappa_score(y_test,y_pred)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Balanced Accuracy:",metrics.balanced_accuracy_score(y_test, y_pred))
#print("Precision:",metrics.precision_score(y_test, y_pred))
#print("Recall:",metrics.recall_score(y_test, y_pred))
#print("F1-score:",metrics.f1_score(y_test, y_pred))
print("MCC:",metrics.matthews_corrcoef(y_test, y_pred))
print('Kappa score:',metrics.cohen_kappa_score(y_test,y_pred))
#fpr6, tpr6, _ = metrics.roc_curve(y_test,  y_pred)
#auc = metrics.roc_auc_score(y_test, y_pred)
#print("AUC:",auc)

f1_micro_RF=metrics.f1_score(y_test, y_pred, average='micro')

FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP)
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
print("This is FPR",FPR)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy for each class
ACC = (TP+TN)/(TP+FP+FN+TN)


f.write("\nFPR :"+str( FPR))
f.write("\nF1-micro"+str(f1_micro_RF))


f.write("\n------------------------------------ RANDOM FOREST ----------------------------------------------------------\n")
f.write(str(confusion_matrix(y_test,y_pred)))
f.write("\nAccuracy:"+ str(metrics.accuracy_score(y_test, y_pred)))
f.write("\nBalanced Accuracy:"+str(metrics.balanced_accuracy_score(y_test, y_pred)))
#f.write("\nPrecision:"+str(metrics.precision_score(y_test, y_pred)))
#f.write("\nRecall:"+str(metrics.recall_score(y_test, y_pred)))
#f.write("\nF1-score:"+str(metrics.f1_score(y_test, y_pred)))
f.write("\nMCC:"+str(metrics.matthews_corrcoef(y_test, y_pred)))
f.write('\nKappa score:'+str(metrics.cohen_kappa_score(y_test,y_pred)))
#f.write("\nAUC:"+str(auc))
df_report_RF=classification_report(y_test,y_pred, output_dict=True)
RF_f1_classes=[]
for i in range(0,5):
    RF_f1_classes.append(df_report_RF[str(i)]['f1-score'])


ax = sns.heatmap(cm, annot=True, cmap='Blues')

ax.set_title('Random Forest Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Category')
ax.set_ylabel('Actual Category ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Benign','Mirai', 'MiTM','DoS','Scanning'])
ax.yaxis.set_ticklabels(['Benign','Mirai', 'MiTM','DoS','Scanning'])
plt.savefig(path_to_file+'RF_cm.png',bbox_inches='tight')
## Display the visualization of the Confusion Matrix.
plt.show()
''''importances = classifier.feature_importances_
std = np.std([tree.feature_importances_ for tree in classifier.estimators_], axis=0)
forest_importances = pd.Series(importances, index=df1.columns[1:len(df1.columns)])

f.close()
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
'''
############################################################# XG BOOST #############################################################
print("------------------------------------XGBOOST------------------------------------------------")
# Training XGBoost on the Training set
from xgboost import XGBClassifier
classifier_XG= XGBClassifier(colsample_bytree=0.6, gamma=0.5,max_depth=5,min_child_weight=1,subsample=0.8,objective= "multi:softmax", num_class=5)
classifier_XG.fit(xtrain, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier_XG.predict(xtest)

print(confusion_matrix(y_test,y_pred))


'''params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

grid_search = GridSearchCV(estimator=classifier_XG,
                           param_grid=params,
                           cv=cv,  verbose=3)



result=grid_search.fit(xtrain,y_train)

best_XG=result.best_estimator_
print('Best Score: %s' % result.best_score_)
print("Best Hyperparameters: %s" % result.best_params_)

y_pred=best_XG.predict(xtest)'''
cm = confusion_matrix(y_test, y_pred)
print(cm)
kappa_XG=metrics.cohen_kappa_score(y_test,y_pred)
f1_score_XG= metrics.f1_score(y_test, y_pred, average='macro')
accuracy_XG=metrics.balanced_accuracy_score(y_test, y_pred)
MCC_XG=metrics.matthews_corrcoef(y_test, y_pred)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Balanced Accuracy:",metrics.balanced_accuracy_score(y_test, y_pred))
#print("Precision:",metrics.precision_score(y_test, y_pred))
#print("Recall:",metrics.recall_score(y_test, y_pred))
#print("F1-score:",metrics.f1_score(y_test, y_pred))
print("MCC:",metrics.matthews_corrcoef(y_test, y_pred))
print('Kappa score:',metrics.cohen_kappa_score(y_test,y_pred))
#fpr8, tpr8, _ = metrics.roc_curve(y_test,  y_pred)
#auc = metrics.roc_auc_score(y_test, y_pred)
#print("AUC:",auc)

#cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
print(classification_report(y_test,y_pred))
df_report_XG=classification_report(y_test,y_pred, output_dict=True)
XG_f1_classes=[]
for i in range(0,5):
    XG_f1_classes.append(df_report_XG[str(i)]['f1-score'])
''''
score= cross_validate(classifier,xtrain,y_train,cv=cv , scoring=_scoring,return_train_score=True)
scores=pd.DataFrame(score)
print(scores)


train_XG_cross_result_f1=scores['train_f1'].mean()*100
test_XG_cross_result_f1=scores['test_f1'].mean()*100
train_XG_cross_result_roc=scores['train_roc_auc'].mean()*100
test_XG_cross_result_roc=scores['test_roc_auc'].mean()*100
train_XG_cross_result_BA=scores['train_balanced_accuracy'].mean()*100
test_XG_cross_result_BA=scores['test_balanced_accuracy'].mean()*100'''



ax = sns.heatmap(cm, annot=True, cmap='Blues')

ax.set_title('XG BOOST Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Category')
ax.set_ylabel('Actual Category ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Benign','Mirai', 'MiTM','DoS','Scanning'])
ax.yaxis.set_ticklabels(['Benign','Mirai', 'MiTM','DoS','Scanning'])
plt.savefig(path_to_file+'XG_cm.png',bbox_inches='tight')
## Display the visualization of the Confusion Matrix.
plt.show()
f1_micro_XG=metrics.f1_score(y_test, y_pred, average='micro')

FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP)
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
print("This is FPR",FPR)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy for each class
ACC = (TP+TN)/(TP+FP+FN+TN)

f.write("\n-----------------------XGBOOST-----------------------------------------------------")
f.write("\nFPR :"+str( FPR))
f.write("\nF1-micro"+str(f1_micro_XG))


###################################################### MAKE GRAPHS ############################################
N = 6
ind = np.arange(N)  # the x locations for the groups
width = 0.15     # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

yvals = [ KNN_f1_classes[0]*100,LR_f1_classes[0]*100, SVM_f1_classes[0]*100,DT_f1_classes[0]*100,RF_f1_classes[0]*100,XG_f1_classes[0]*100]
rects1 = ax.bar(ind, yvals, width, color='cornflowerblue')
zvals = [ KNN_f1_classes[1]*100,LR_f1_classes[1]*100, SVM_f1_classes[1]*100,DT_f1_classes[1]*100,RF_f1_classes[1]*100,XG_f1_classes[1]*100]
rects2 = ax.bar(ind+width, zvals, width, color='firebrick')
kvals = [KNN_f1_classes[2]*100,LR_f1_classes[2]*100,SVM_f1_classes[2]*100,DT_f1_classes[2]*100,RF_f1_classes[2]*100,XG_f1_classes[2]*100]
rects3 = ax.bar(ind+width*2, kvals, width, color='yellowgreen')

zvals = [ KNN_f1_classes[3]*100,LR_f1_classes[3]*100, SVM_f1_classes[3]*100,DT_f1_classes[3]*100,RF_f1_classes[3]*100,XG_f1_classes[3]*100]
rects4 = ax.bar(ind+width+width*2, zvals, width, color='mediumpurple')
kvals = [KNN_f1_classes[4]*100,LR_f1_classes[4]*100,SVM_f1_classes[4]*100,DT_f1_classes[4]*100,RF_f1_classes[4]*100,XG_f1_classes[4]*100]
rects5 = ax.bar(ind+width+width+width*2, kvals, width, color='lightblue')

ax.set_title('F1_score per class')
ax.set_ylabel("f1-score %")
x=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
ax.set_xticks(ind+width/2)
ax.set_xticklabels( ( 'KNN', 'Logistic Regression', 'SVM','Decision Tree','Random Forest','XGBOOST') )
ax.legend( (rects1[0], rects2[0], rects3[0],rects4[0],rects5[0]), ('Normal','Mirai','MiTM','DoS','Scanning') ,loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)

def autolabel(rects):
    for rect in rects:

        h = rect.get_height()
        if h==100:
            ax.text(rect.get_x()+rect.get_width()/2., 1*h,int(h),
                ha='center', va='bottom')
        else:
            ax.text(rect.get_x() + rect.get_width() / 2., 1 * h, "{:.1f}".format(h),
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)
plt.show()







N = 6
ind = np.arange(N)  # the x locations for the groups
width = 0.15      # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

yvals = [ f1_score_KNN*100,f1_score_LR*100, f1_score_SVM*100,f1_score_DT*100,f1_score_RF*100,f1_score_XG*100]
rects1 = ax.bar(ind, yvals, width, color='cornflowerblue')
zvals = [accuracy_KNN*100,accuracy_LR*100, accuracy_SVM*100,accuracy_DT*100,accuracy_RF*100,accuracy_XG*100]
rects2 = ax.bar(ind+width, zvals, width,  color='firebrick')
''''kvals = [MCC_KNN,MCC_LR,MCC_SVM,MCC_DT,MCC_RF]
rects3 = ax.bar(ind+width*2, kvals, width, color='b')'''

''''mvals = [KNN_f1_classes,LR_f1_classes,SVM_f1_classes,DT_f1_classes,RF_f1_classes]
rects4 = ax.bar(ind+width+width*2, mvals, width, color='y')'''

ax.set_xticks(ind+width/2)
ax.set_xticklabels( ( 'KNN', 'Logistic Regression', 'SVM','Decision Tree','Random Forest','XG BOOST') )
ax.legend( (rects1[0], rects2[0]), ('F1_score macro', 'Balanced_accuracy') )

def autolabel(rects):
    for rect in rects:

        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1*h, '%.2f'%(h),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()

########################################################### KAPPA SCORE ###########################################



list=['KNN', 'Logistic Regression', 'SVM','Decision Tree','Random Forest',"XGBOOST"]

kvals = [kappa_KNN,kappa_LR,kappa_SVM,kappa_DT,kappa_RF,kappa_XG]


y_pos = np.arange(len(list))
threshold=0
# Basic plot
plt.bar(y_pos, kvals, color=(0.2, 0.4, 0.6, 0.6),width=0.25)
#plt.plot([0, len(list)], [threshold, threshold], "k--")
#plt.ylim(-1,1,1.0)
plt.ylabel("kappa score")
plt.title("Cohen's Kappa score")
# use the plt.xticks function to custom labels
plt.xticks(y_pos, list)
#plt.legend()
plt.show()
