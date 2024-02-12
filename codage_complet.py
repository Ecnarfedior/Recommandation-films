import netrc
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns




#chargement des bases
title_akas = pd.read_csv('title.akas.tsv',sep = "\t")
name_basic = pd.read_csv('name.basic.tsv',sep = "\t")
title_basic = pd.read_csv('title.basics.tsv',sep = "\t")
title_crew = pd.read_csv('title.crew.tsv',sep = "\t")
title_episode = pd.read_csv('title.episode.tsv',sep = "\t")
title_principal = pd.read_csv('title.principal.tsv',sep = "\t")
title_rating = pd.read_csv('title.rating.tsv',sep = "\t")
tmdb_full = pd.read_csv('tmdb_full.csv')


#exploration des bases
title_akas.isnull().sum()
title_akas.head(10)
name_basic.isnull().sum()
name_basic.head(10)
title_crew.isnull().sum()
title_crew.head(10)
title_episode.isnull().sum()
title_episode.head(10)
title_principal.isnull().sum()
title_principal.head(10)
title_rating.isnull().sum()
title_rating.head(10)
tmdb_full.isnull().sum()
tmdb_full.head(10)


#compte tenue du postulat, soit être missionné pour une première approche des possibilité du data driven 
# et lancer une application de recommandation de film. Nous avons identifié sur le fichier title_basic la colonne titleType identifiant les styles des tconst. 
# Nous avons donc fait le choix de filtrer en gardant uniquement les type Movie et inner join les fichiers utiles pour le df final. 
# A noter que les fichiers Akas, Episode et Crew ont été exclue du tableau final.


# filtrage sur title basic pour servir de base de rattachement en inner afin d'exclure les lignes n'étant pas en movie
basic = title_basic
basic_movie_test = basic[basic['titleType'].str.contains('movie')]
basic_movie_test.to_csv('basic_movie_test')

#création des fichiers allégé
basic_filtre_movie = pd.read_csv('basic_movie_test')

#renomage de colonne pour join
title_akas.rename(columns={'titleId':'tconst'}, inplace = True)
join_basic_akas = basic_filtre_movie.merge(right=title_akas,on='tconst',how='inner')
join_basic_akas.to_csv('join_basic_akas')

join_basic_principal = basic_filtre_movie.merge(right=title_principal,on='tconst',how='inner')
join_basic_principal.to_csv('join_basic_principal')

join_basic_crew = basic_filtre_movie.merge(right=title_crew,on='tconst',how='inner')
join_basic_crew.to_csv('join_basic_crew')

join_basic_rating = basic_filtre_movie.merge(right=title_rating,on='tconst',how='inner')
join_basic_rating.to_csv('join_basic_rating')

tmdb_full['release_date'] = pd.to_datetime(tmdb_full['release_date'])
tmdb_full.rename(columns={'imdb_id':'tconst'}, inplace = True)
join_basic_tmdb = basic_filtre_movie.merge(right=tmdb_full,on='tconst',how='inner')
join_basic_tmdb.to_csv('join_basic_tmdb')

# les fichiers ont été nettoyé des colonnes jugé inutiles dans le cadre de la mission par un drop['colonne'], axis = 1

# merge globale des fichiers
rating=pd.read_csv('df_rating')
name_basic=pd.read_csv('merge_namebasic_dfprincipal')
tmdb = pd.read_csv('join_basic_tmdb')

# nettoyage des fichiers
tmdb = tmdb.drop('Unnamed: 0',axis=1)
tmdb = tmdb.drop('Unnamed: 0.1',axis=1)
rating = rating.drop('Unnamed: 0',axis=1)
name_basic=name_basic.drop('Unnamed: 0',axis=1)
tmdb=tmdb.drop('isAdult',axis=1)
tmdb=tmdb.drop('endYear',axis=1)
tmdb=tmdb.drop('runtimeMinutes',axis=1)
tmdb=tmdb.drop('title',axis=1)
tmdb=tmdb.drop('adult',axis=1)
tmdb=tmdb.drop('genres_y',axis=1)
tmdb=tmdb.drop('id',axis=1)
tmdb=tmdb.drop('original_title',axis=1)
tmdb=tmdb.drop('video',axis=1)
tmdb=tmdb.drop('production_companies_name',axis=1)
tmdb=tmdb.drop('production_companies_country',axis=1)

df1 = tmdb.merge(right=name_basic,on='tconst',how='inner')
df2 =df1.merge(right=rating,on='tconst',how='inner')
merge_final = df2
merge_final['primaryProfession'] = merge_final['primaryProfession'].fillna('actor')
merge_final = merge_final.to_csv('Merge Final')

# travail sur le dataframe en vue de son utilisation global (KPI, corrélation et Machine learning)
a_bosser=pd.read_csv('Merge Final')
a_bosser = a_bosser.drop('Unnamed: 0',axis=1)

# création des colonnes portant sur les genres 
# ( à noter que les genres sont au maximum de 3 dans une cellule et classé par ordre alphabétique)
a_bosser['genre_1']=a_bosser['genres_x'].str.split(',').str[0].str.strip()
a_bosser['genre_2']=a_bosser['genres_x'].str.split(',').str[1].str.strip()
a_bosser['genre_3']=a_bosser['genres_x'].str.split(',').str[2].str.strip()
a_bosser = a_bosser.drop('genres_x',axis=1)

# même travail sur les professions
a_bosser['profession_1']=a_bosser['primaryProfession'].str.split(',').str[0].str.strip()
a_bosser['profession_2']=a_bosser['primaryProfession'].str.split(',').str[1].str.strip()
a_bosser['profession_3']=a_bosser['primaryProfession'].str.split(',').str[2].str.strip()
a_bosser = a_bosser.drop('primaryProfession',axis=1)

a_bosser = a_bosser.drop('startYear',axis=1)

# filtre appliqué sur les durées de films, décision prise de garder uniquement les films entre 45 minutes et 4 heures
reduit = a_bosser.drop(a_bosser.query('runtime>240').index)
reduit = reduit.drop(reduit.query('runtime<45').index)

# travail sur les dates
testdate = reduit
testdate['year']=testdate['release_date'].str.split('-').str[0].str.strip()
testdate['mois']=testdate['release_date'].str.split('-').str[1].str.strip()
testdate['jour']=testdate['release_date'].str.split('-').str[2].str.strip()

# travail sur les valeurs du df
testdrop2 = testdate.dropna(subset=['year'])
testdrop2.isna().sum()
testdrop2['genre_2'].fillna('no_genre_2', inplace = True)
testdrop2['genre_3'].fillna('no_genre_3', inplace = True)
testdrop2['profession_2'].fillna('no_prof_2', inplace = True)
testdrop2['profession_3'].fillna('no_prof_3', inplace = True)
testdrop3 = testdrop2[testdrop2['year'] != 1]
testdrop4 = testdrop3[testdrop3['year'] != 5]
finalna = testdrop4
finalna['tagline'].fillna('no_tagline', inplace = True)
finalna['poster_path'].fillna('no_poster_path', inplace = True)
finalna['overview'].fillna('no_overview', inplace = True)
finalna['homepage'].fillna('no_homepage', inplace = True)
finalna['backdrop_path'].fillna('no_backdrop_path', inplace = True)
finalna['release_date']=pd.to_datetime(finalna['release_date'])

# impression du dataframe en csv
ter=finalna.to_csv('pret pour dummies')

# préparation du df pour exploitation
globale =pd.read_csv('pret pour dummies')
nepastoucher = globale

# création d'intervalle d'années pour les films, car après un dummies des années et des mois, 
# nous avons estimé que cela n'était pas judicieux pour l'exploitation
groupyear = globale[['tconst','year']]

# choix fait de 8 intervalles passant par un filtrage et des df intermédiaires
inter1 = groupyear[groupyear['year']<1961]
inter1['intervalle_1900_1960'] = 1

inter2 = groupyear[groupyear['year']<1981]
inter2 = inter2[inter2['year']>1960]
inter2['intervalle_1960_1980']=1

inter3 = groupyear[groupyear['year']<1991]
inter3 = inter3[inter3['year']>1980]
inter3['intervalle_1980_1990']=1

inter4 = groupyear[groupyear['year']<2001]
inter4 = inter4[inter4['year']>1990]
inter4['intervalle_1990_2000']=1

inter5 = groupyear[groupyear['year']<2011]
inter5 = inter5[inter5['year']>2000]
inter5['intervalle_2000_2010']=1

inter6 = groupyear[groupyear['year']<2016]
inter6 = inter6[inter6['year']>2010]
inter6['intervalle_2010_2015']=1

inter7 = groupyear[groupyear['year']<2021]
inter7 = inter7[inter7['year']>2015]
inter7['intervalle_2015_2020']=1

inter8 = groupyear[groupyear['year']>2020]
inter8['intervalle_2020_et+']=1

# regroupage des df intermédiaire par un concat, les intervalles avant 1980 n'ont pas été conservé
df_inter2 = pd.concat([inter3,inter4,inter5,inter6,inter7,inter8], axis=0)

# remplacement des valeurs null en 0
df_inter2 = df_inter2.fillna(0)

# mise au format int64 des colonnes
df_inter2['intervalle_1980_1990'] = df_inter2['intervalle_1980_1990'].astype(np.int64)
df_inter2['intervalle_1990_2000'] = df_inter2['intervalle_1990_2000'].astype(np.int64)
df_inter2['intervalle_2000_2010'] = df_inter2['intervalle_2000_2010'].astype(np.int64)
df_inter2['intervalle_2010_2015'] = df_inter2['intervalle_2010_2015'].astype(np.int64)
df_inter2['intervalle_2015_2020'] = df_inter2['intervalle_2015_2020'].astype(np.int64)
df_inter2['intervalle_2020_et+'] = df_inter2['intervalle_2020_et+'].astype(np.int64)

# rattachement des colonnes d'intervalles au df principal
df_intervalle4 = globale.merge(right=df_inter2, on= ['tconst','year'], how = 'inner')

# ce join a généré une multiplication des lignes que nous avons identifier et traité
df_intervalle4=df_intervalle4.drop_duplicates()

# nettoyage des genres non renseigné
nettoyage = df_intervalle4
nettoyage['genre_1']=nettoyage['genre_1'].replace('\\N', '0')
nettoyage = nettoyage.drop(nettoyage.query('genre_1=="0"').index)

# netoyage de quelques colonnes non exploité
nettoyage = nettoyage.drop('Unnamed: 0', axis=1)
nettoyage = nettoyage.drop('titleType', axis=1)
nettoyage = nettoyage.drop('ordering', axis=1)

# impression en csv
base = nettoyage.to_csv('nettoyee 1')

# finalisation du dataframe pour exploitation
df = pd.read_csv('nettoyee 1')

# travail sur les genres car le dummies n'est pas concluant et exploitable

#identification des genres
df['genre_1'].unique() 

# mise au format liste
listeGenre = ['Drama', 'Comedy', 'Action', 'Documentary', 'Horror', 'Crime',
       'Sci-Fi', 'Biography', 'Adventure', 'Thriller', 'Romance',
       'Fantasy', 'Adult', 'Family', 'Animation', 'Musical', 'Western',
       'Mystery', 'History', 'Music', 'War', 'Sport', 'Reality-TV',
       'News']
       
# création d'une boucle qui parcours la liste, pour chaque éléments on créer une colonne du même nom 
# et si dans une des 3 colonnes genre l'élément est présent alors on implante 1 dans la cellule sinon 0
for i in listeGenre:
    df[i] = np.where((df['genre_1']== i)|(df['genre_2']== i)|(df['genre_3']== i), 1, 0)
    
# nettoyage de colonnes
df = df.drop('genre_1',axis=1)
df = df.drop('genre_2',axis=1)
df = df.drop('genre_3',axis=1)
df = df.drop('Unnamed: 0',axis=1)
df = df.drop('jour',axis=1)
df['release_date']= pd.to_datetime(df['release_date'])

# création de colonnes sur certains éléments afin de pouvoir les utiliser dans le Machine learning et plus spécifiquement améliorer le voisinage
df['fact_homepage']= np.where(df['homepage']== 'no_homepage', 0, 1)
df['fact_backdrop_path']= np.where(df['backdrop_path']== 'no_backdrop_path', 0, 1)
df['fact_poster_path']= np.where(df['poster_path']== 'no_poster_path', 0, 1)
df['fact_overview']= np.where(df['overview']== 'no_overview', 0, 1)
df['fact_tagline']= np.where(df['tagline']== 'no_tagline', 0, 1)

# nettoyage des colonnes vote et note
df['vote_average']= df['vote_average'].apply(lambda x : -3 if x <1 else x)
df['vote_count']= df['vote_count'].apply(lambda x : 1 if x <1 else x)

# création d'une colonne ratio entre les notes et le nombre de vote
df['Ratio_Vcount/Vavg']=round((df['vote_count']/df['vote_average'])/10,2)
df['Ratio_Vcount/Vavg'].fillna(0, inplace=True)

# création d'une colonne opinion basé sur le ratio
colonne_cible = 'Ratio_Vcount/Vavg'
intervalles = [0, 0.5, 1.5, 6, 500]
categories = ['Peu fiable', 'Moyennement fiable', 'Fiable', 'Tres fiable']
df['Opinion'] = pd.cut(df[colonne_cible], bins=intervalles, labels=categories, include_lowest=True)

df = pd.to_csv('Base_prete')

import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Chargement des données avec cache
@st.cache_data
def load_data():
    link = './pages/Base_prete.csv'
    df = pd.read_csv(link)
    return df

df = load_data()

# Sélectionner les caractéristiques numériques
numeric_features = df.select_dtypes(include='number').columns.tolist()

# Liste des films à prédire
films_list = [
   'Levante', 'Bên trong vo kén vàng', 'Halkara', 'Tiger Stripes', 'Greatest Days', 'The Beast Beneath',
    'Rock Hudson: All That Heaven Allowed', 'Translations', 'May December', 'Anthropophagus II',
    'Last Summer of Nathan Lee', 'Bolan\'s Shoes', 'L\'été dernier', 'Liuben', 'Asog', 'Mojave Diamonds',
    'Anatomie d\'une chute', 'Öte', 'Les Jours heureux', 'Buddy Games: Spring Awakening', 'A través del mar',
    'Laissez-moi', 'Flo', 'Il pleut dans la maison', 'Mountains', 'The Seeding', 'Black Clover: Sword of the Wizard King',
    'Vincent doit mourir', 'Kitty the Killer', 'Kötü Adamin 10 Günü', 'Inshallah walad', 'Downtown Owl',
    'La fille de son père', 'Gehen und Bleiben', 'L\'autre Laurens', 'Killer Kites', 'Puentes en el mar',
    'Enter the Clones of Bruce', 'Eric Clapton: Across 24 Nights', 'In Flames', 'Autumn Moon', 'Lost Soulz',
    'Le ravissement', 'Skad dokad', 'La mer et ses vagues', 'The Future', 'Q', 'All Up in the Biz',
    'Taylor Mac\'s 24-Decade History of Popular Music', 'One Night with Adela', 'Nattevagten - Dæmoner går i arv',
    'Lost Country', 'Kimitachi wa dô ikiru ka', 'Have You Got It Yet? The Story of Syd Barrett...',
    'The Country Club', 'Outlaw Johnny Black', 'Days of Daisy'
]

# Prédictions de popularité pour les films
predicted_popularity_list = []
df_released = df[df['status'] == 'Released']
X_released = df_released[numeric_features]
y_released = df_released['popularity']

model = LinearRegression()
model.fit(X_released, y_released)

for film_title in films_list:
    data = df[df['originalTitle'] == film_title]
    X_data = data[numeric_features]
    
    # Faire des prédictions si des données existent pour ce film
    if not X_data.empty:
        predictions = model.predict(X_data)
        predicted_popularity_list.append((film_title, predictions[0]))

# Trier les prédictions pour obtenir les 5 meilleures
top_5_predicted_popularity = sorted(predicted_popularity_list, key=lambda x: x[1], reverse=True)[:5]

# Afficher les 5 meilleurs résultats de popularité prédits
#('Top 5 des films prédits en popularité')
#for rank, (film_title, popularity_prediction) in enumerate(top_5_predicted_popularity, start=1):
    #(f"Numero #{rank}: {film_title} ----- Popularité prédite: {popularity_prediction}")

# Graphique des 10 meilleurs films prédits en popularité
st.title('Top 10 des films en production prédits en popularité ')
fig, ax = plt.subplots(figsize=(8, 6))

# Obtenir les données pour le graphique
top_10_predicted_popularity = sorted(predicted_popularity_list, key=lambda x: x[1], reverse=True)[:10]
film_titles = [film[0] for film in top_10_predicted_popularity]
popularity_predictions = [film[1] for film in top_10_predicted_popularity]

# Créer le diagramme à barres
ax.bar(film_titles, popularity_predictions, color='red')
plt.xticks(rotation=90)

# Ajouter des titres et libellés au graphique
ax.set_facecolor('white')
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
plt.ylabel('Popularité prédite', color='black')
plt.title('Top 10 des films prédits en popularité', color='black')

# Afficher le graphique dans Streamlit
st.pyplot(fig)

import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Chargement des données depuis le fichier CSV
link ='./pages/Base_prete.csv'
df1 = pd.read_csv(link)

# Sélection des colonnes incluant des nombres
features = df1.select_dtypes(include=['number']).columns.tolist()

# Groupement des données par titre de film
grouped = df1.groupby('originalTitle')[features].mean().reset_index()

# Normalisation
scaler = StandardScaler()
scaled_features = scaler.fit_transform(grouped[features])

# KNN
k = 10  # Nombre de voisins
model = NearestNeighbors(n_neighbors=k, metric='euclidean')
model.fit(scaled_features)

# Fonction de recommandation des films
def recommend_movies(movie_title, num_recommendations=5):
    movie_indices = grouped[grouped['originalTitle'] == movie_title].index.values
    if len(movie_indices) == 0:
        print("Le film spécifié n'est pas trouvé.")
        return []
    
    movie_index = movie_indices[0] 
    distances, indices = model.kneighbors([scaled_features[movie_index]])
    recommended_movies = []
    for i in range(1, num_recommendations + 1):
        recommended_movies.append(grouped.iloc[indices[0][i]]['originalTitle'])
    return recommended_movies

# Ajouter les URLs complètes des affiches dans le DataFrame
def get_full_poster_url(poster_path):
    base_url = 'https://image.tmdb.org/t/p/w500/'
    return base_url + poster_path

df1['poster_url'] = df1['poster_path'].apply(get_full_poster_url)

# Interface Streamlit
st.title("Système de recommandation de films")
film_utilisateur = st.text_input("Entrez le nom d'un film :", "Inception")  # Valeur par défaut : "Inception"

if st.button("Obtenir les recommandations"):
    recommended_movies = recommend_movies(film_utilisateur)

    if recommended_movies:
        st.write(f"Pour '{film_utilisateur}', les films recommandés sont :")
        
        # Mise en page en ligne des images des affiches
        cols = st.columns(len(recommended_movies))
        for i, movie_title in enumerate(recommended_movies):
            poster_url = df1[df1['originalTitle'] == movie_title]['poster_url'].iloc[0]
            with cols[i]:
                st.image(poster_url, caption=movie_title, width=150)
    else:
        st.write("Le film spécifié n'a pas été trouvé.")

import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@st.cache_data
def load_data():
    link = './pages/Base_prete.csv'
    df = pd.read_csv(link)
    return df

@st.cache_data
def preprocess_data(df):
    features = df.select_dtypes(include=['number']).columns.tolist()
    grouped = df.groupby('originalTitle')[features].mean().reset_index()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(grouped[features])

    return grouped, scaled_features

def recommend_movies(grouped, scaled_features, movies):
    k = 5
    model = NearestNeighbors(n_neighbors=k, metric='euclidean')
    model.fit(scaled_features)

    movie_indices = []
    for movie in movies:
        indices = grouped[grouped['originalTitle'] == movie].index.values
        while len(indices) == 0:
            st.warning(f"Le film '{movie}' spécifié n'est pas trouvé.")
            movie = st.text_input(f"Entrez à nouveau le titre du film '{movie}':")
            indices = grouped[grouped['originalTitle'] == movie].index.values
        movie_indices.extend(indices)
    
    movie_indices = list(set(movie_indices))  # Supprimer les doublons
    recommendations = []
    for index in movie_indices:
        distances, indices = model.kneighbors([scaled_features[index]])
        recommended_movies = []
        for i in range(1, 2):  # Obtenir un seul film recommandé pour chaque entrée
            recommended_movies.append(grouped.iloc[indices[0][i]]['originalTitle'])
        recommendations.extend(recommended_movies)
    return recommendations

def get_full_poster_url(poster_path):
    base_url = 'https://image.tmdb.org/t/p/w500/'
    return base_url + poster_path

def main():
    st.title("Recommandation de films")
    df = load_data()
    grouped, scaled_features = preprocess_data(df)

    movies = []
    for i in range(3):
        movie_input = st.text_input(f"Entrez le film {i + 1}:", "Inception")
        movies.append(movie_input)

    if st.button("Obtenir une recommandation"):
        recommended_movies = recommend_movies(grouped, scaled_features, movies)
        st.write("Résultat de la recommandation basée sur les trois films:")
        st.write(recommended_movies[0])  # Afficher la recommandation unique
       
    
        # Obtenir l'URL de l'affiche
        poster_url = df[df['originalTitle'] == recommended_movies[0]]['poster_path'].iloc[0]
        full_poster_url = get_full_poster_url(poster_url)
        st.image(full_poster_url, caption=recommended_movies[0], width=250)
        

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd


# Chargement des données depuis le fichier CSV avec cache
@st.cache_data
def load_data():
    link = './pages/Base_prete.csv'
    df1 = pd.read_csv(link)
    return df1
   
def get_full_poster_url(poster_path):
    base_url = 'https://image.tmdb.org/t/p/w500/'
    return base_url + poster_path

# Interface Streamlit
st.title("Recommandations de films basées sur les acteurs")
actor_name = st.text_input("Entrez le nom d'un acteur :", "Brad Pitt")  # Valeur par défaut : "Brad Pitt"

# Filtrage des films en fonction du nom de l'acteur
df1 = load_data()  # Chargement des données
filtered_movies = df1[(df1['profession_1'] == 'actor') | 
                      (df1['profession_2'] == 'actor') |
                      (df1['profession_3'] == 'actor')]
filtered_movies = filtered_movies[filtered_movies['primaryName'] == actor_name]

# Affichage des films associés à l'acteur en colonnes de 4
if not filtered_movies.empty:
    st.write(f"Les films associés à l'acteur '{actor_name}' sont :")
    cols = st.columns(4)
    for index, movie in filtered_movies.iterrows():
        poster_url = get_full_poster_url(movie['poster_path'])
        with cols[int(index % 4)]:
            st.image(poster_url, caption=movie['originalTitle'], width=150)
else:
    st.write(f"Aucun film associé à l'acteur '{actor_name}' trouvé.")

import streamlit as st
import pandas as pd


# Chargement des données avec cache
@st.cache_data
def load_data():
    link = './pages/Base_prete.csv'
    df1 = pd.read_csv(link)
    return df1

df1 = load_data()

def get_full_poster_url(poster_path):
    base_url = 'https://image.tmdb.org/t/p/w500/'
    return base_url + poster_path

# Options disponibles pour les années et les genres
available_years = ['1980_1990', '1990_2000', '2000_2010', '2010_2015', '2015_2020', '2020_et+']
available_genres = ['Drama', 'Comedy', 'Action', 'Documentary', 'Horror', 'Crime', 'Sci-Fi', 'Biography',
          'Adventure', 'Thriller', 'Romance', 'Fantasy', 'Family', 'Animation', 'Musical',
          'Western', 'Mystery', 'History', 'Music', 'War', 'Sport', 'Reality-TV', 'News']

# Interface Streamlit
st.title("Recherche de films par année et genre")

# Sélection de l'année et du genre
selected_year = st.selectbox("Sélectionner une intervalle d'année :", available_years)
selected_genre = st.selectbox("Sélectionner un genre :", available_genres)

# Bouton pour lancer la recherche
if st.button("Rechercher"):
    # Filtrage des films en fonction de l'année et du genre sélectionnés
    filtered_movies = df1[(df1[f'intervalle_{selected_year}'] == 1) & (df1[selected_genre] == 1)]
    
    # Éliminer les doublons selon le nom du film
    unique_movies = filtered_movies.drop_duplicates(subset='originalTitle')

    sorted_movies = unique_movies.sort_values(by='vote_count', ascending=False)
    
    # Tri des films uniques par vote_count et sélection des 15 films les plus votés
    top_movies = unique_movies.sort_values(by='vote_count', ascending=False).head(16)
    
    # Affichage des films par groupe de quatre côte à côte
    if not top_movies.empty:
        st.write(f"Les 15 films uniques les mieux notés pour l'année '{selected_year}' et le genre '{selected_genre}' sont :")
        
        cols = st.columns(4)  # Divise l'affichage en 4 colonnes
        
        count = 0
        for index, movie in top_movies.iterrows():
            if count % 4 == 0:
                cols = st.columns(4)  # Réinitialise les colonnes pour chaque groupe de quatre films
            
            with cols[count % 4]:
                poster_url = get_full_poster_url(movie['poster_path'])
                try:
                    st.image(poster_url, caption=movie['originalTitle'], width=150)
                except Exception as e:
                    st.error(f"Impossible de charger l'image : {e}")
            
            count += 1
    else:
        st.write("Aucun film trouvé pour cette sélection.")

import streamlit as st
import pandas as pd

link = './pages/Base_prete.csv'
df = pd.read_csv(link)

# Fonction pour obtenir l'URL complète de l'affiche
def get_full_poster_url(poster_path):
    base_url = 'https://image.tmdb.org/t/p/w500/'
    return base_url + poster_path

df['poster_url'] = df['poster_path'].apply(get_full_poster_url)

# Liste des genres
genres = ['Drama', 'Comedy', 'Action', 'Documentary', 'Horror', 'Crime', 'Sci-Fi', 'Biography',
          'Adventure', 'Thriller', 'Romance', 'Fantasy', 'Family', 'Animation', 'Musical',
          'Western', 'Mystery', 'History', 'Music', 'War', 'Sport', 'Reality-TV', 'News']

top_movies_by_genre = {}

# Extraire les meilleurs films par genre
for genre in genres:
    genre_data = df[df[genre] == 1]
    grouped_genre_data = genre_data.groupby('originalTitle').first().reset_index()
    top_movies_genre = grouped_genre_data.sort_values(by='vote_count', ascending=False).head(5)
    top_movies_by_genre[genre] = top_movies_genre

# Interface Streamlit
st.title("Les 5 meilleurs films par genre")

# Affichage des films par genre
for genre, top_movies in top_movies_by_genre.items():
    st.header(f"Genre: {genre}")
    
    # Calcul du nombre de colonnes
    num_columns = min(5, len(top_movies))  # On prend soit 5 soit le nombre de films disponibles
    
    # Création des colonnes pour les affiches
    cols = st.columns(num_columns)
    for i in range(num_columns):
        if i < len(top_movies):
            movie = top_movies.iloc[i]
            poster_url = movie['poster_url']
            
            with cols[i]:
                st.image(poster_url, caption=movie['originalTitle'], width=150)

