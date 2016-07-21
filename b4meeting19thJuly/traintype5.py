# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 17:20:08 2016

Learning started
Prediction started
Results : rbf Kernel
[[ 1.          0.49117651]
 [ 0.49117651  1.        ]]
SpearmanrResult(correlation=0.50654597490973974, pvalue=0.0)


@author: ajay
"""
import sys
import os
os.chdir("/home/ajay/MTP")
import numpy as np
from scipy.stats.stats import spearmanr, pearsonr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
from matplotlib import style
style.use("ggplot")
from sklearn import svm, preprocessing
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier


#Regression Module Testing
def Build_Data_Set(features = ["citation2yrs","RDI","rcount","auth_prod_avg","authprod_max","auth_div_avg","auth_div_max","auth_hindex_avg","auth_hindex_max","auth_soc_avg","auth_soc_max","team"]):
    citation_data = pd.DataFrame.from_csv("./workspace/SVR.txt")
    citation_data = citation_data.loc[citation_data['paper_year'] <= 1995]   
    citation_data.iloc[np.random.permutation(len(citation_data))]
    
    X = np.array(citation_data[features].values)
    #X = preprocessing.scale(X)
    y = (citation_data["paper_cat"].values.tolist())
    return X,y

X, y = Build_Data_Set()

clf = OneVsOneClassifier(SVC(random_state=0, kernel='rbf'))
print("Learning started")
clf.fit(X,y)
#Saving the classifier
import pickle

save_classifier = open("./workspace/rbfsvr.pickle","wb")
pickle.dump(clf, save_classifier)
save_classifier.close()


#testing
def Build_Data_Set(features = ["citation2yrs","RDI","rcount","auth_prod_avg","authprod_max","auth_div_avg","auth_div_max","auth_hindex_avg","auth_hindex_max","auth_soc_avg","auth_soc_max","team"]):
    citation_data = pd.DataFrame.from_csv("./workspace/SVR.txt")
    citation_data = citation_data.loc[citation_data['paper_year'] >= 1996]   
    citation_data.iloc[np.random.permutation(len(citation_data))]
    
    X = np.array(citation_data[features].values)
    #X = preprocessing.scale(X)
    y = (citation_data["paper_cat"].values.tolist())
    return X,y
test_X, test_y = Build_Data_Set()

"""
classifier_f = open("./workspace/rbfsvr.pickle", "rb")
clf= pickle.load(classifier_f)
classifier_f.close()
"""
print("Prediction started")
prediction = clf.predict(test_X)

from sklearn.metrics import confusion_matrix
confusion_matrix(test_y, prediction)

pearson = np.corrcoef(prediction, test_y)
spearman = spearmanr(prediction, test_y)

print("Results : rbf Kernel ")
print(pearson)
print(spearman)




"""
citation_data = pd.DataFrame.from_csv("./workspace/merged_major_features_all.txt")
citation_data1 = citation_data.loc[citation_data['paper_cat'] == 1] 
citation_data1 = citation_data.loc[citation_data['paper_year'] <= 2005]
spearmanr(citation_data1['citation5yrs'], citation_data1['countX'])
#SpearmanrResult(correlation=0.27798290778996548, pvalue=6.6521695082463583e-157)

citation_data1 = citation_data.loc[citation_data['paper_cat'] == 2] 
citation_data1 = citation_data1.loc[citation_data1['paper_year'] >= 1996]
spearmanr(citation_data1['citation5yrs'], citation_data1['countX'])
#SpearmanrResult(correlation=0.22512583609623329, pvalue=0.0)

citation_data1 = citation_data.loc[citation_data['paper_cat'] == 3] 
citation_data1 = citation_data1.loc[citation_data1['paper_year'] >= 1996]
spearmanr(citation_data1['citation5yrs'], citation_data1['countX'])
#SpearmanrResult(correlation=0.42155403745806325, pvalue=0.0)

citation_data1 = citation_data.loc[citation_data['paper_cat'] == 4] 
citation_data1 = citation_data1.loc[citation_data1['paper_year'] >= 1996]
spearmanr(citation_data1['citation5yrs'], citation_data1['countX'])
#SpearmanrResult(correlation=0.067286389707734914, pvalue=3.8776542167950394e-09)

citation_data1 = citation_data.loc[citation_data['paper_cat'] == 5] 
citation_data1 = citation_data1.loc[citation_data1['paper_year'] >= 1996]
spearmanr(citation_data1['citation5yrs'], citation_data1['countX'])
#SpearmanrResult(correlation=0.3341618225906362, pvalue=8.3552333377338231e-119)
"""

