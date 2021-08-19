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

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as pt

imdb = pd.read_csv('imdb_df.csv')

imdb.isnull().sum(axis = 0)
imdb.head()

imdb.describe()


