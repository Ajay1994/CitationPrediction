# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 23:31:56 2016

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

"""
out = open("/home/ajay/code/years_final.txt")
paper_yr_dict = dict()
for line in out:
    try:
        paper_id = int(line.split(":")[0])
        paper_year = int(line.split(":")[1])
        paper_yr_dict[paper_id] = paper_year
    except:
        continue
out.close()

out = open("/home/ajay/code/years.txt")
paper_yr_dict_left = dict()
for line in out:
    try:
        paper_id = int(line.split(":")[0])
        paper_year = int(line.split(":")[1])
        paper_yr_dict_left[paper_id] = paper_year
    except:
        continue
out.close()

paper_yr_dict.update(paper_yr_dict_left)

import pickle

save_year_dic = open("./data/final_data_set/paper_year_dict.pickle","wb")
pickle.dump(paper_yr_dict, save_year_dic)
save_year_dic.close()
"""

fout = open("./workspace/citation_579_year_count_extended_year.txt")
fin = open("./workspace/citation_579_year_count_category_extended_year.txt" , "w")
fin.write("paper_id,paper_year,citation2yrs,citation5yrs,citation7yrs,citation9yrs,category\n")
for line in fout:
    try:
        paper_id = int(line.split(",")[0])
        category = paper_cat_dict[paper_id]
        fin.write(line.rstrip('\n')+","+str(category) +"\n")
    except:
        continue
fin.close()
fout.close()

############################################
#Regression Module Testing
def Build_Data_Set(features = ["citation2yrs","authprod_max",'auth_prod_avg']):
    citation_data = pd.DataFrame.from_csv("./workspace/merged_major_features_all_with_category.txt")
    citation_data = citation_data.loc[citation_data['paper_cat'] == 3]   
    citation_data.iloc[np.random.permutation(len(citation_data))]
    
    X = np.array(citation_data[features].values)
    #X = preprocessing.scale(X)
    y = (citation_data["citation5yrs"].values.tolist())
    return X,y



test_size = 3000
X, y = Build_Data_Set()

    
clf = svm.SVC(kernel="poly", C= 1.0)
clf.fit(X[:-test_size],y[:-test_size])

truth = y[-test_size:]
predicted = clf.predict(X[-test_size:])

print(np.corrcoef(truth, predicted))
print(spearmanr(truth, predicted))
plt.scatter(truth, predicted)


#################################################
#LIBSVM Method