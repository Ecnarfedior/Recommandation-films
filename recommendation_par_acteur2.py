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
