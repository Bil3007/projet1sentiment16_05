# sentiment_analysis_app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


# Entraînement du modèle
def train_model():
    
    vectorizer = joblib.load('vectorizer.pkl')
    
    model = joblib.load('Best_mod.pkl')
    
    return model, vectorizer

def main():
    st.title("Analyse de sentiment avec Streamlit")
    
    
    # Entraîner le modèle
    model, vectorizer = train_model()
    
        
    # Interface utilisateur pour tester le modèle
    user_input = st.text_input("Entrez votre avis sur un film :")
    
    if st.button("Analyser"):
        # Vectorisation du texte de l'utilisateur
        user_input_vec = vectorizer.transform([user_input])
        
        # Prédiction du sentiment
        prediction = model.predict(user_input_vec)[0]
        
        if prediction == 4:
            st.write("Sentiment : Positif")
        else:
            st.write("Sentiment : Négatif")

if __name__ == "__main__":
    main()
