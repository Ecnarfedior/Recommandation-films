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
