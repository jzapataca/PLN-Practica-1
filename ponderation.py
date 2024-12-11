import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import pandas as pd

def calculate_to(corpus):
    """
    Calcula la frecuencia absoluta (TO) de las características.
    """
    vectorizer = CountVectorizer()
    to_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    
    return to_matrix, feature_names, vectorizer

def calculate_tfidf(corpus):
    """
    Calcula la ponderación TF-IDF de las características.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    
    return tfidf_matrix, feature_names, vectorizer

def calculate_ponderation():
    """
    Calcula la frecuencia absoluta (TO) y la ponderación TF-IDF de las características para el corpus
    del momento en el que se ejecute la función. Además guarda las matrices y los objetos vectorizer en archivos.
    """
    # Cargar el corpus desde el archivo CSV
    df = pd.read_csv("data/processed/general_corpus.csv")
    corpus = df["Text"].tolist()

    # Calcular la frecuencia absoluta (TO)
    to_matrix, to_feature_names, to_vectorizer = calculate_to(corpus)

    # Calcular la ponderación TF-IDF
    tfidf_matrix, tfidf_feature_names, tfidf_vectorizer = calculate_tfidf(corpus)

    # Guardar las matrices y los nombres de las características
    np.save("data/processed/ponderation/to_matrix.npy", to_matrix.toarray())
    np.save("data/processed/ponderation/to_feature_names.npy", to_feature_names)

    np.save("data/processed/ponderation/tfidf_matrix.npy", tfidf_matrix.toarray())
    np.save("data/processed/ponderation/tfidf_feature_names.npy", tfidf_feature_names)

    # Guardar los objetos vectorizer
    with open("data/processed/ponderation/to_vectorizer.pkl", "wb") as file:
        pickle.dump(to_vectorizer, file)

    with open("data/processed/ponderation/tfidf_vectorizer.pkl", "wb") as file:
        pickle.dump(tfidf_vectorizer, file)

    print("Matrices y objetos vectorizer guardados exitosamente.")

if __name__ == '__main__':
    calculate_ponderation()