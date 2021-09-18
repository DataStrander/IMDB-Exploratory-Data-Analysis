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
from pandas.io.formats import style
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.express as px

imdb = pd.read_csv('imdb_df.csv')

imdb.isnull().sum(axis = 0)

imdb = imdb.drop(columns=['primaryTitle','Unnamed: 0','knownForTitles','directors','writers','isAdult','primaryProfession'])
imdb = imdb.rename(columns={'titleType':'type','originalTitle':'title','primaryName':'director'})
imdb.head()

# +
'''Data Cleansing'''

print(imdb['type'].unique(), imdb.info())


# +
'''Genre Analysis'''

imdb_list = imdb['genres'].values.tolist()
imdb[['main_genres']] = imdb['genres'].astype(str).str.split(",",expand=True,).get(0)
# for i in range(imdb["genres"].str.split(",", n = 1, expand = True)): data["Name_{}".format()]= new[i]
# for genre in imdb_list:
#    f, ax = plt.subplots(figsize=(10, 6))
#    sns.countplot(x='startYear', data=imdb[df], palette="Greens_d");
#    plt.title(genre)

# +
'''Genres Production'''

genres = imdb.groupby("main_genres")['main_genres'].count().nlargest(30)
ind = genres.index # the index(row labels) of the dataframe

fig = px.bar(imdb, x = genres, y = ind, color = ind)
# fig = go.Figure()
# fig.add_trace(go.Bar(x = genres, y = ind))
fig.update_layout(title='Title Amount by Genre', xaxis_title='Movie Count', yaxis_title='Genres',
                 width=1000, height=400)
fig.show()
# -



# +
'''Genres Rating Cluster'''

genre_filter = imdb.groupby('main_genres').filter(lambda x : len(x)>5).sort_values(by="startYear")
#the filter() method subsets the dataframe rows or columns according to the specified index labels.
g = sns.lineplot(data=genre_filter, x=genre_filter["startYear"], y=genre_filter['averageRating'], hue=genre_filter["startYear "],
                 ci=None,linewidth = 2, palette="Set2")
g.legend(loc='upper right', bbox_to_anchor=(1.35, 1), prop={'size': 15})
fig = plt.gcf()
fig.set_size_inches(12, 8)
# -

fig = go.Figure(data=go.Scatter(x=imdb['averageRating'], y=imdb['startYear'], size=len(imdb['type']),
                               color=imdb['genres'], hover_name="type", log_x=True, size_max=60))
fig.update_layout(plot_bgcolor=imdb["genres"], paper_bgcolor=colors["genres"], font_color=colors["type"])
fig.show()




