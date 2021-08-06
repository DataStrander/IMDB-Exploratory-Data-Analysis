# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import gzip
import shutil
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as pt

# +
'''Zipping TSV Loading'''

# Gzipping just one file from our raw directory
with gzip.open('./raw_data/name.basics.tsv.gz', 'rb') as f_in:
    with open('./raw_data/name_basics.txt', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
title_akas = pd.read_csv('./raw_data/name_basics.txt', sep='\t')
# -

title_akas.head()

# +
'''Data Processing Fucntion on Multiple Files'''

# Assign path
files = next(os.walk("./raw_data/"))
# Assign dataset names
list_of_names = ['title.akas','title.basics','title.crew','title.episodes','title.principals','title.raitings']
# Create empty list to store the multiple files we want to open
dataframes_list = []
  
# append datasets into teh list
for i in range(len(list_of_names)):
    with gzip.open("./raw_data/"+list_of_names[i]+'.tsv.gz', 'rb') as f_in:
        with open("./raw_data/"+list_of_names[i]+'.txt', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    temp_df = pd.read_csv("./raw_data/"+list_of_names[i]+".txt", sep='\t')
    dataframes_list.append(temp_df)
# -


