import pandas as pd
import numpy as np
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
pd.set_option('display.max_columns', None)

df = pd.read_csv("C:/Users/DELL/Downloads/IMDB_Top250Engmovies2_OMDB_Detailed.csv")
df.head()

len(df)

#DATA PREPROCESSING


# convert lowercase and remove numbers, punctuations, spaces, etc.,
df['clean_plot'] = df['plot'].str.lower()
df['clean_plot'] = df['clean_plot'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
df['clean_plot'] = df['clean_plot'].apply(lambda x: re.sub('\s+', ' ', x))
df['clean_plot']

# tokenize the sentence
df['clean_plot'] = df['clean_plot'].apply(lambda x: nltk.word_tokenize(x))
df['clean_plot']

# remove stopwords
stop_words = nltk.corpus.stopwords.words('english')
plot = []
for sentence in df['clean_plot']:
    temp = []
    for word in sentence:
        if word not in stop_words and len(word) >= 3:
            temp.append(word)
    plot.append(temp)
plot
df['clean_plot'] = plot
df['clean_plot']
df.head()
def clean(sentence):
    temp = []
    for word in sentence:
        temp.append(word.lower().replace(' ', ''))
    return temp
df['genre'] = [clean(x) for x in df['genre']]
df['actors'] = [clean(x) for x in df['actors']]
df['director'] = [clean(x) for x in df['director']]

def clean(sentence):
    temp = []
    for word in sentence:
        temp.append(word.lower().replace(' ', ''))
    return temp
df['genre'] = [clean(x) for x in df['genre']]
df['actors'] = [clean(x) for x in df['actors']]
df['director'] = [clean(x) for x in df['director']]
df['actors'][0]

# combining all the columns data
columns = ['clean_plot', 'genre', 'actors', 'director']
l = []
for i in range(len(df)):
    words = ''
    for col in columns:
        words += ' '.join(df[col][i]) + ' '
    l.append(words)
l

df['clean_input'] = l
df = df[['title', 'clean_input']]
df.head()

#FEATURE EXTRACTION

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
tfidf = TfidfVectorizer()
features = tfidf.fit_transform(df['clean_input'])


# create cosine similarity matrix
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(features, features)
print(cosine_sim)

#MOVIE RECOMMENDATION

index = pd.Series(df['title'])
index.head()

def recommend_movies(title):
    movies = []
    idx = index[index == title].index[0]
    # print(idx)
    score = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
    top10 = list(score.iloc[1:11].index)
    # print(top10)
    
    for i in top10:
        movies.append(df['title'][i])
    return movies

recommend_movies('The Dark Knight Rises')

index[index == 'The Dark Knight Rises'].index[0]

pd.Series(cosine_sim[3]).sort_values(ascending=False)
recommend_movies('The Shawshank Redemption')
recommend_movies('The Avengers')