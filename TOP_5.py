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
