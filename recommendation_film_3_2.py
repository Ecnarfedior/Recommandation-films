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
