from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def calculate_to(corpus):
    """
    Calcula la frecuencia absoluta (TO) de las características.
    """
    vectorizer = CountVectorizer()
    to_matrix = vectorizer.fit_transform([' '.join(tokens) for tokens in corpus])
    feature_names = vectorizer.get_feature_names_out()
    
    return to_matrix, feature_names

def calculate_tfidf(corpus):
    """
    Calcula la ponderación TF-IDF de las características.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([' '.join(tokens) for tokens in corpus])
    feature_names = vectorizer.get_feature_names_out()
    
    return tfidf_matrix, feature_names

if __name__ == '__main__':

    # Cargar el corpus
    with open("data/processed/general_corpus.txt", "r", encoding="utf-8") as file:
        corpus = [line.strip().split() for line in file]

    # Calcular la frecuencia absoluta (TO)
    to_matrix, feature_names = calculate_to(corpus)

    # Calcular la ponderación TF-IDF
    tfidf_matrix, feature_names = calculate_tfidf(corpus)

    print("Matriz de frecuencia absoluta (TO):")
    print(to_matrix)

    print("Matriz de ponderación TF-IDF:")
    print(tfidf_matrix)

    