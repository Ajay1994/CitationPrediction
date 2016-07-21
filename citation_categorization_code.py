import sys
import os
os.chdir("/home/ajay/MTP/")
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
import pickle


'''
###########################################################################
print("Started Making Years Dict ...")
fout = open("/home/ajay/MTP/data/final_data_set/years.txt")
paper_year_dict = dict()
count = 0
for line in fout:
    try:
        paper_id = int(line.split(":")[0])
        paper_year = int(line.split(":")[1])
        paper_year_dict[paper_id] = paper_year
    except:
        #print(line)
        count += 1
fout.close()

fout = open("/home/ajay/MTP/vikashcode/index_years")
paper_year_dict_rest = dict()
count = 0
for line in fout:
    try:
        paper_id = int(line.split(":")[0])
        paper_year = int(line.split(":")[1])
        paper_year_dict_rest[paper_id] = paper_year
    except:
        print(line)
        count += 1
fout.close()

z = paper_year_dict.copy()
paper_year_dict.update(paper_year_dict_rest)
print("Years Dictionary Completed ...")

###################################################################################
'''


fout = open("./data/citation_network.txt")
fin = open("./workspace/citer_cited_year.txt", "w")
count = 0
for line in fout:
    try:
        paper_citer = int(line.split("\t")[0])
        paper_cited = int(line.split("\t")[1])
        fin.write(str(paper_citer) + "\t" + str(paper_cited) + "\t" + str(paper_year_dict[paper_citer]) + "\n")
        fin.flush()
    except:
        #print(line)
        count += 1
fout.close()
fin.close()

####################################################################################
fin = open("./workspace/all_paper.txt", "w")
for paper in z:
    fin.write(str(paper) + "\n")
    fin.flush()
fin.close()

fin = open("./workspace/all_paper_year.txt", "w")
for paper in z:
    fin.write(str(paper) + "\t"+str(z[paper])+"\n")
    fin.flush()
fin.close()
#####################################################################################

"""
print("Dictionary Completed ... Saving ")
save_citation_network = open("./workspace/citation_network_dict","wb")
pickle.dump(citer_cited_dict, save_citation_network)
save_citation_network.close()
"""
print("Dirty data", count)
print("Done")

"""

print("Started Making Years Dict ...")
fout = open("/home/ajay/MTP/data/final_data_set/years.txt")
paper_year_dict = dict()
count = 0
for line in fout:
    try:
	paper_id, paper_year = map(int, line.rstrip().split(':'))
        paper_year_dict[paper_id] = paper_year
    except:
        #print(line)
        count += 1
fout.close()



network_dict = open("./workspace/citation_network_dict", "rb")
citer_cited_dict= pickle.load(network_dict)
network_dict.close()
"""
fout = open("/home/ajay/MTP/workspace/citation_network.txt")
nodes_within_dict = dict()
count = 0
for line in fout:
    try:
        paper_citer = int(line.split("\t")[0])
        paper_cited = int(line.split("\t")[1])
        if paper_citer in paper_year_dict:
            nodes_within_dict[paper_citer] = paper_year_dict[paper_citer]
        if paper_cited in paper_year_dict:
            nodes_within_dict[paper_cited] = paper_year_dict[paper_cited]
    except:
        #print(line)
        count += 1
fout.close()

wdin1970to95 = 0
wdin1996to00 = 0
for paper in nodes_within_dict:
    if nodes_within_dict[paper] >= 1970 and nodes_within_dict[paper] <= 1995:
        wdin1970to95 += 1
    if nodes_within_dict[paper] >= 1996 and nodes_within_dict[paper] <= 2000:
        wdin1996to00 += 1
print("1970 - 95", wdin1970to95)
print("1996 - 00", wdin1996to00)
