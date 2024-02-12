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
