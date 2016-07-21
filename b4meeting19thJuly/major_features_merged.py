# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 15:04:12 2016

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

#Creating of the citation Dictionary
paper_2yrs_dict = dict()
paper_5yrs_dict = dict()
paper_7yrs_dict = dict()
paper_9yrs_dict = dict()
paper_year_dict = dict()
fout = open("./data/feature_data/all_network_paper_CitationProfile.txt")
for line in fout:
    try:
        paper_id = int(line.split("\t")[0])
        paper_year = int(line.split("\t")[1])
        paper_citation = line.split("\t")[2]
        citations = paper_citation.split(",")
        citation_2yrs = "NA"
        citation_5yrs = "NA"
        citation_7yrs = "NA"
        citation_9yrs = "NA"
        if len(citations) >= 3:
            citation_2yrs = (int(citations[0]) + int(citations[1]) + int(citations[2]))
        if len(citations) >= 6:
            citation_5yrs = int(citations[5])
        if len(citations) >= 8:
            citation_7yrs = int(citations[7])
        if len(citations) >= 10:
            citation_9yrs = int(citations[9])
        paper_2yrs_dict[paper_id] = citation_2yrs
        paper_5yrs_dict[paper_id] = citation_5yrs
        paper_7yrs_dict[paper_id] = citation_7yrs
        paper_9yrs_dict[paper_id] = citation_9yrs
        paper_year_dict[paper_id] = paper_year
    except:
        #print(paper_id)
        continue
fout.close()

#Creation of the countX Dictioary
fout = open("/home/ajay/MTP/data/final_data_set/countX_raw_data.txt")
lineno = 0
paper_countX_dict = dict()
for line in fout:
    try:
        if lineno % 5 == 0:
            paper_id = int(line.split(":")[1].strip())
        if lineno % 5 == 3:
            countXList = line.split(":")[1].strip()
            countXList = countXList.split(",")
        if lineno % 5 == 4:
            countX = 0
            numZeros = 0
            for i in list(range(0,3)):
                if countXList[i] == '0':
                    numZeros += 1
            if numZeros != 3:
                countX = (float(countXList[0]) + float(countXList[1]) + float(countXList[2]))/(3 - numZeros)
        lineno += 1
        paper_countX_dict[paper_id] = countX
    except:
        continue
fout.close()


#Creation of the citewords Dictioary
fout = open("/home/ajay/MTP/data/final_data_set/citewords_raw_data.txt")
lineno = 0
paper_citewords_dict = dict()
for line in fout:
    try:
        if lineno % 5 == 0:
            paper_id = int(line.split(":")[1].strip())
        if lineno % 5 == 3:
            citewordsList = line.split(":")[1].strip()
            citewordsList = citewordsList.split(",")
        if lineno % 5 == 4:
            citeword = 0.0
            numZeros = 0
            for i in list(range(0,3)):
                if citewordsList[i] == '0':
                    numZeros += 1
            if numZeros != 3:
                citeword = (float(citewordsList[0]) + float(citewordsList[1]) + float(citewordsList[2]))/(3 - numZeros)
        lineno += 1
        paper_citewords_dict[paper_id] = citeword
    except:
        continue
fout.close()

#Creation of the category Dictioary
out = open("./data/final_data_set/categories.txt")
paper_cat_dict = dict()
for line in out:
    try:
        paper_id = int(line.split(",")[0])
        paper_cat = int(line.split(",")[1])
        paper_cat_dict[paper_id] = paper_cat
    except:
        continue
out.close()

#Creation of the RDI Dictioary
out = open("./data/all_paper_reference_diversity.txt")
paper_RDI_dict = dict()
for line in out:
    try:
        paper_id = int(line.split(",")[0])
        paper_RDI = float(line.split(",")[1])
        paper_RDI_dict[paper_id] = paper_RDI
    except:
        continue
out.close()

#Creation of the Reference Count Dictioary
out = open("./data/all_paper_reference_count.txt")
paper_refCount_dict = dict()
for line in out:
    try:
        paper_id = int(line.split(",")[0])
        paper_rcount = int(line.split(",")[1])
        paper_refCount_dict[paper_id] = paper_rcount
    except:
        continue
out.close()

#Creation of the author productivity Dictioary
out = open("./data/feature_data/all_network_paper_author_max_avg_prominence.txt")
paper_auth_dict = dict()
for line in out:
    try:
        paper_id = int(line.split("\t")[0])
        paper_prod_max = (line.split("\t")[1]).strip("\n")
        paper_prod_avg = (line.split("\t")[2]).strip("\n")
        paper_auth_dict[paper_id] = paper_prod_avg+","+paper_prod_max
    except:
        continue
out.close()


#Creation of the author Diversity Dictioary
out = open("./data/feature_data/all_network_paper_author_max_avg_diversity.txt")
paper_authdiv_dict = dict()
for line in out:
    try:
        paper_id = int(line.split("\t")[0])
        paper_div_max = (line.split("\t")[1]).strip("\n")
        paper_div_avg = (line.split("\t")[2]).strip("\n")
        paper_authdiv_dict[paper_id] = paper_div_avg+","+paper_div_max
    except:
        continue
out.close()


#Creation of the author h-index Dictioary
out = open("./data/feature_data/all_network_paper_author_max_avg_h_index.txt")
paper_hindex_dict = dict()
for line in out:
    try:
        paper_id = int(line.split("\t")[0])
        paper_hindex_max = (line.split("\t")[1]).strip("\n")
        paper_hindex_avg = (line.split("\t")[2]).strip("\n")
        paper_hindex_dict[paper_id] = paper_hindex_avg+","+paper_hindex_max
    except:
        continue
out.close()

#Creation of the author socilaity Dictioary
out = open("./data/feature_data/all_network_paper_author_max_avg_co_author.txt")
paper_social_dict = dict()
for line in out:
    try:
        paper_id = int(line.split("\t")[0])
        paper_social_max = (line.split("\t")[1]).strip("\n")
        paper_social_avg = (line.split("\t")[2]).strip("\n")
        paper_social_dict[paper_id] = paper_social_avg+","+paper_social_max
    except:
        continue
out.close()


#Creation of the team size Dictioary
out = open("./data/feature_data/all_network_paper_author_count.txt")
paper_team_dict = dict()
for line in out:
    try:
        paper_id = int(line.split("\t")[0])
        paper_team_max = (line.split("\t")[1]).strip("\n")
        paper_team_dict[paper_id] = paper_team_max
    except:
        continue
out.close()

#Merging of all the papers of the dump(without_category) 
fin = open("./workspace/merged_major_features_all.txt", "w")
fin.write("paper_id,paper_year,citation2yrs,citation5yrs,citation7yrs,citation9yrs,countX,citeword,RDI,rcount,auth_prod_avg,authprod_max,auth_div_avg,auth_div_max,auth_hindex_avg,auth_hindex_max,auth_soc_avg,auth_soc_max,team\n")
for paper in paper_year_dict:
    if paper in paper_citewords_dict:
        try:
            fin.write(str(paper) + ","+ str(paper_year_dict[paper]) + ","+ str(paper_2yrs_dict[paper]) + ","+ str(paper_5yrs_dict[paper]) + ","+ str(paper_7yrs_dict[paper]) + ","+ str(paper_9yrs_dict[paper]) + ","+ str(paper_countX_dict[paper]) + ","+ str(paper_citewords_dict[paper]) + ","+ str(paper_RDI_dict[paper]) + ","+ str(paper_refCount_dict[paper]) + ","+ str(paper_auth_dict[paper]) + ","+ str(paper_hindex_dict[paper]) + ","+ str(paper_social_dict[paper]) + ","+ str(paper_team_dict[paper])+"\n")
        except:
            print(paper)
            continue
    fin.flush()
fin.close()

#Merging of all the papers of the dump(with_category) 
fin = open("./workspace/merged_major_features_all_with_category.txt", "w")
fin.write("paper_id,paper_year,paper_cat,citation2yrs,citation5yrs,citation7yrs,citation9yrs,countX,citeword,RDI,rcount,auth_prod_avg,authprod_max,auth_div_avg,auth_div_max,auth_hindex_avg,auth_hindex_max,auth_soc_avg,auth_soc_max,team\n")
for paper in paper_year_dict:
    if paper in paper_citewords_dict and paper in paper_cat_dict:
        try:
            fin.write(str(paper) + ","+ str(paper_year_dict[paper]) + ","+ str(paper_cat_dict[paper]) + ","+ str(paper_2yrs_dict[paper]) + ","+ str(paper_5yrs_dict[paper]) + ","+ str(paper_7yrs_dict[paper]) + ","+ str(paper_9yrs_dict[paper]) + ","+ str(paper_countX_dict[paper]) + ","+ str(paper_citewords_dict[paper]) + ","+ str(paper_RDI_dict[paper]) + ","+ str(paper_refCount_dict[paper]) + ","+ str(paper_auth_dict[paper]) + ","+ str(paper_authdiv_dict[paper]) + ","+ str(paper_hindex_dict[paper]) + ","+ str(paper_social_dict[paper]) + ","+ str(paper_team_dict[paper])+"\n")
        except:
            continue
    fin.flush()
fin.close()

#SVR TRAINNING
fin = open("./workspace/SVR.txt", "w")
fin.write("paper_id,paper_year,paper_cat,citation2yrs,RDI,rcount,auth_prod_avg,authprod_max,auth_div_avg,auth_div_max,auth_hindex_avg,auth_hindex_max,auth_soc_avg,auth_soc_max,team\n")
for paper in paper_year_dict:
    if paper in paper_cat_dict:
        try:
            fin.write(str(paper) + ","+ str(paper_year_dict[paper]) + ","+ str(paper_cat_dict[paper]) + ","+ str(paper_2yrs_dict[paper]) + ","+ str(paper_RDI_dict[paper]) + ","+ str(paper_refCount_dict[paper]) + ","+ str(paper_auth_dict[paper]) + ","+ str(paper_authdiv_dict[paper]) + ","+ str(paper_hindex_dict[paper]) + ","+ str(paper_social_dict[paper]) + ","+ str(paper_team_dict[paper])+"\n")
        except:
            continue
    fin.flush()
fin.close()


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

clf = OneVsOneClassifier(SVC(random_state=0, kernel='poly'))
clf.fit(X,y)
#Saving the classifier
import pickle

save_classifier = open("./workspace/polysvr.pickle","wb")
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
prediction = clf.predict(test_X)
