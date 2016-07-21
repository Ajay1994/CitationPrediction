# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:35:00 2016

@author: ajay
"""

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

fout = open("./workspace/categories.txt")
cat_dict = dict()
count = 0
for line in fout:
    try:
        #print(line)
        paper = int(line.split(",")[0])
        paper_cat = int(line.split(",")[1])
        cat_dict[paper] = paper_cat
    except:
        print(line)
        count += 1
fout.close()


fout = open("./workspace/categories_tanmoy.txt")
cat_dict1 = dict()
count = 0
for line in fout:
    try:
        #print(line)
        paper = int(line.split("\t")[0])
        paper_cat = int(line.split("\t")[1])
        cat_dict1[paper] = paper_cat
    except:
        print(line)
        count += 1
fout.close()


matched = 0
notMatched = 0
for paper in cat_dict:
    if paper in cat_dict1:
        if cat_dict[paper] == cat_dict1[paper]:
            matched += 1
        else:
            notMatched += 1