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
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash_html_components.Label import Label
from dash.dependencies import Input, Output

imdb = pd.read_csv('imdb_df.csv')

imdb.isnull().sum(axis = 0)

imdb = imdb.drop(columns=['primaryTitle','Unnamed: 0','knownForTitles','directors','writers','isAdult','primaryProfession'])
imdb = imdb.rename(columns={'titleType':'type','originalTitle':'title','primaryName':'director'})
imdb.head()

# +
'''Data Cleansing'''

print(imdb['type'].unique())


# +
'''Genre Analysis'''

seperate_genre=['Action','Adventure','Animation','Biography','Comedy','Crime','Drama','Fantasy','Family',
                'History','Horror','Music','Musical','Mystery','Romance','Sci-Fi','Sport','Thriller','War','Western']

for genre in seperate_genre:
    df = imdbdata['genres'].str.contains(genre).fillna(False)
    print('The total number of movies with ',genre,'=',len(imdbdata[df]))
    f, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Year', data=imdbdata[df], palette="Greens_d");
    plt.title(genre)
    compare_movies_rating = ['Runtime_Minutes', 'Votes','Revenue_Millions', 'Metascore']
    for compare in compare_movies_rating:
        sns.jointplot(x='Rating', y=compare, data=imdbdata[df], alpha=0.7, color='b', size=8)
        plt.title(genre)
# -

fig = go.Figure(data=go.Scatter(x=imdb['averageRating'], y=imdb['startYear'], size=len(imdb['type']),
                               color=imdb['genres'], hover_name="type", log_x=True, size_max=60))
fig.update_layout(plot_bgcolor=imdb["genres"], paper_bgcolor=colors["genres"], font_color=colors["type"])
fig.show()




