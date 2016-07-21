# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 11:28:51 2016

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
from math import log

fout = open("./data/citation_network_small.txt")
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


fout = open("./data/fields_citations")
referecne_type = dict()
for line in fout:
    paper = line.split("\t")[0].strip()
    ref_type = line.split("\t")[1].strip()
    referecne_type[paper] = ref_type
fout.close()

fout = open("./data/all_paper_reference_diversity.txt", "w")
fout.write("paper,RDI\n")
for paper in referecne_count_dict:
    references = referecne_count_dict[paper]
    areas = []
    for referecne in references:
        try:
            areas.append(referecne_type[referecne])
        except:
            continue
    total_ref = len(areas)
    if total_ref == 0:
        continue
    unique_ref = set(areas)
    entropy = 0
    for element in unique_ref:
        count = areas.count(element)
        entropy = entropy -(float(count)/total_ref)*log(float(count)/total_ref,2)
    fout.write(str(paper)+","+str(entropy))
    fout.write("\n")
fout.close()