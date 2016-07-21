# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 22:26:52 2016

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

fout = open("./data/citation_network.txt")
referecne_count_dict = dict()
for line in fout:
    citer = line.split("\t")[0].strip()
    cited = line.split("\t")[1].strip()
    if citer not in referecne_count_dict:
        referecne_count_dict[citer] = []
        referecne_count_dict[citer].append(cited)
    else:
        referecne_count_dict[citer].append(cited)
fout.close()
        
fout = open("./data/reference_count.txt", "w")
fout.write("paper,ref_count\n")
for citer in referecne_count_dict:
    fout.write(str(citer)+","+str(len(referecne_count_dict[citer])))
    fout.write("\n")
fout.close()
