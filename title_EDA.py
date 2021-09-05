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

from imdb import IMDb
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go

imdb = pd.read_csv('imdb_df.csv')

imdb.isnull().sum(axis = 0)

imdb = imdb.drop(columns=['primaryTitle','Unnamed: 0','knownForTitles','directors','writers','isAdult','primaryProfession'])
imdb = imdb.rename(columns={'titleType':'type','originalTitle':'title','primaryName':'director'})
imdb.head()

# +
'''Data Cleansing'''

print(imdb['type'].unique())

# -

fig = go.Figure(data=go.Scatter(x=imdb['averageRating'], y=imdb['startYear']))
fig.show()


