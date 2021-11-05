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
import nltk
from nltk.corpus import names
from nltk import NaiveBayesClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.express as px

imdb = pd.read_csv('imdb_df.csv')

imdb.isnull().sum(axis = 0)

imdb = imdb.drop(columns=['primaryTitle','Unnamed: 0','knownForTitles','directors','writers','isAdult','primaryProfession'])
imdb = imdb.rename(columns={'titleType':'type','originalTitle':'title','primaryName':'director'})
imdb.dropna(how='any')
imdb.head()

# +
'''Data Cleansing'''

print(imdb['type'].unique(), imdb.info())


# +
'''Genre Analysis'''

imdb_list = imdb['genres'].to_numpy().tolist()
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

# +
'''Director Box Plot'''

director = imdb.groupby("director")['type','averageRating','startYear'].mean()
director = director.nlargest(8, 'averageRating')

# Boxplot to see the outlaier trend for each director
fig = px.box(director, x=director.index, y="averageRating", color="averageRating",
             notched=True, # used notched shape
             hover_data=["averageRating"]
            )
fig.update_layout(title='Most Appreciated Directors', title_font_family='Open Sans'
                  xaxis_title='Director', yaxis_title='Best Rating', width=1000, height=400)
fig.show()
# +
'''Movie Appreciation Time Series'''

fig = go.Figure()
fig.add_trace(go.Scatter(x=imdb['averageRating'], y=imdb['startYear'], mode='markers', # size=len(imdb['type']),
                         marker=dict(color='LightSkyBlue',size=120), line=dict(color='MediumPurple', width=12),
                         # color=imdb['genres'], hover_name="type", size_max=60)
                         showlegend=False)
             )
fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='rgba(0,0,0,0)')
fig.show()

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
fig = go.Figure()
for contestant, group in imdb.groupby("Contestant"):
    fig.add_trace(go.Bar(x=group["Fruit"], y=group["Number Eaten"], name=contestant))
fig.update_layout(legend_title_text = "Best Movie Director by Film Rating")
fig.update_xaxes(title_text="Director")
fig.update_yaxes(title_text="Rating")
fig.show()

# +
'''Native Bayes Classifier for Character Classification'''

# Let's split the tile to have a list of it and perform a Dataframe construction on the two files
titles = titles.split('\n')
# Unfortunately \n is still remaining but we can easily remove it
plots = plots.replace('\n', ' ')
print(len(titles),len(plots))

# Now it is time to wrap up the two files in an a cleaned dataframe in order to classify which titles have female protagonist
df = pd.DataFrame({'title':titles, 'plot':plots})
df['words'] = df['plot'].str.split().str.len()
df['plot'] = df['plot'].astype(str)

# It is also possible to compare the name inside each plots with a dataset of names and see how frequently the name is occurring in the plot
web_names = pd.read_csv('https://query.data.world/s/wvx63qksvakxjpp4elvqwva45scuk3')
web_names = web_names.rename(columns={'John':'name'})

# Before diving inside the real NLP, we can simply see if there are any general gender proposition inside the text that are recurring the most 
# i.e. she/he and her/him. In this way we would be able to find the main character following the common sense
prop = {'male':[' he ',' his ',' him '],'female':[' she ',' her ']}
male = [' he ',' his ',' him ']
female = [' she ',' her ']
words = lambda x: len(x["plot"].split(" ")) -1
df['male'] = (df['plot'].str.count(' he ') + df['plot'].str.count(' his ') + df['plot'].str.count(' him '))
df['female'] = (df['plot'].str.count(' she ') + df['plot'].str.count(' her '))

# Let's also separate what we consider name from the starting words of each sentence
def first_name(text):
    result = re.findall('([A-Z]([a-z])+)', text)
    return " ".join(map(str, result))
df['characters']=df['plot'].apply(lambda x : first_name(x))

# sum(df['male'] > df['female'] for df['female'] in len(df))
print("The number of stories with a female protagonist is {}".format(len(df[df['female']>df['male']])))
df

'''NLTK Classifier -  Not Ultimated'''

# According to an article, most of the female names end with specific letters and therefore we need to build a function that can recognise which letters
# are most likely to belonging to names ending with them -> https://pubmed.ncbi.nlm.nih.gov/11026389/
def gender_features(word):
    return {'last_letter':word[-1]}

# It comes in our help the nltk library with a dataset of all the names available that we can compare with the ones available in the plot
gender_names = ([(name, 'male') for name in names.words('male.txt')]+ [(name, 'female') for name in names.words('female.txt')])
random.shuffle(gender_names)
  
# List comprehension to extract and process the right features for our Gender Classifier
featuresets = [(gender_features(n), gender) for (n, gender)in gender_names]

# Training and Test Sets Prep
train_set, test_set = featuresets[500:], featuresets[:500]
  
# The training set is used to train our Naive Bayes Classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(classifier.classify(gender_features('Olivia')))
# Accuracy of Classifier on Train Set
print(nltk.classify.accuracy(classifier, train_set))
# -




