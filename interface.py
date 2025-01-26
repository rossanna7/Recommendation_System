import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.losses import MeanSquaredError

# Charger et encoder les données
@st.cache_resource
def load_data_and_encoders():
    # Charger le dataset fusionné (celui avec les utilisateurs et les livres)
    df = pd.read_csv('final.csv')  # Assurez-vous que ce fichier est le résultat de votre fusion
    user_encoder = pd.read_pickle("user_encoder.pkl")  # Mettre le chemin correct
    book_encoder = pd.read_pickle("book_encoder.pkl")  # Mettre le chemin correct

    return df, user_encoder, book_encoder

df, user_encoder, book_encoder = load_data_and_encoders()

# Charger le modèle
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('book_recommender_model.h5', custom_objects={'mse': MeanSquaredError()})
    return model

model = load_model()

# Interface Streamlit
st.title("Book Recommendation System")
st.write("Enter your preferences to get personalized book recommendations!")

# Sélectionner des livres à évaluer
selected_books = st.multiselect("Select books to rate:", df['title'].unique())

# Dictionnaire pour stocker les évaluations des livres sélectionnés
ratings = {}

# Permettre à l'utilisateur de donner une note pour chaque livre sélectionné
for book in selected_books:
    rating = st.slider(f"Rate the book '{book}':", min_value=1, max_value=5, step=1)
    ratings[book] = rating

# Générer les recommandations lorsque l'utilisateur appuie sur le bouton
# Générer les recommandations lorsque l'utilisateur appuie sur le bouton
if st.button("Get Recommendations"):
    # Récupérer les IDs des livres sélectionnés et leurs évaluations
    rated_book_ids = df[df['title'].isin(ratings.keys())]['book_id'].values
    rated_ratings = np.array(list(ratings.values()))

    # Générer les prédictions pour les livres non encore évalués
    all_book_ids = df[~df['book_id_encoded'].isin(rated_book_ids)]['book_id_encoded'].unique()  # Exclure les livres déjà notés
    user_tensor = tf.constant([user_encoder.transform([user_id])[0]] * len(all_book_ids))  # ID de l'utilisateur réel
    book_tensor = tf.constant(all_book_ids)

    # Calculer les prédictions
    predictions = model([user_tensor, book_tensor]).numpy().flatten()

    # Trier les livres selon les prédictions et récupérer les meilleurs
    top_indices = predictions.argsort()[-5:][::-1]  # Top 5 livres recommandés
    top_book_ids = all_book_ids[top_indices]
    recommended_books = df[df['book_id_encoded'].isin(top_book_ids)][['title', 'authors']].drop_duplicates()

    # Afficher les livres recommandés
    st.subheader("Recommended Books:")
    for _, row in recommended_books.iterrows():
        st.write(f"**{row['title']}** by {row['authors']}")

