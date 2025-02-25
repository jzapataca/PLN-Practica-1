import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
import string

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')

def tokenize_text(corpus):
    """
    Tokeniza el texto en unigramas y bigramas.
    """
    tokenized_corpus = []
    for text in corpus:
        
        # Tokenización
        tokens = word_tokenize(text.lower())
        
        # Eliminar puntuación
        tokens = [token for token in tokens if token not in string.punctuation]
        
        # Generar unigramas y bigramas
        unigrams = tokens
        
        bigrams = [' '.join(gram) for gram in ngrams(tokens, 2)]

        tokenized_corpus.append(unigrams + bigrams)

    return tokenized_corpus

def remove_stopwords(tokenized_corpus, language='spanish'):
    """
    Elimina stopwords usando la lista de palabras vacías de NLTK.
    """
    
    stop_words = set(stopwords.words(language))
    
    cleaned_corpus = [
        [token for token in tokens if token not in stop_words]
        for tokens in tokenized_corpus
    ]
    
    return cleaned_corpus

def remove_low_frequency_tokens(tokenized_corpus, min_frequency = 3 ):
    """
    Elimina tokens con frecuencia menor al umbral establecido.
    """

    # Contar frecuencias de los tokens
    all_tokens = [token for tokens in tokenized_corpus for token in tokens]
    token_counts = Counter(all_tokens)
    
    # Filtrar tokens según la frecuencia
    cleaned_corpus = [
        [token for token in tokens if token_counts[token] >= min_frequency]
        for tokens in tokenized_corpus
    ]

    return cleaned_corpus

def apply_stemming(tokenized_corpus, language='spanish'):
    """
    Aplica stemming a los tokens.
    """

    stemmer = SnowballStemmer(language)
    
    stemmed_corpus = [
        [stemmer.stem(token) for token in tokens]
        for tokens in tokenized_corpus
    ]
    
    return stemmed_corpus

def feature_extraction(corpus, if_remove_stopwords=True, if_apply_stemming=True, language='spanish'):
    """
    Función principal que ejecuta todo el flujo de procesamiento.
    """
    # Paso 1: Tokenización en unigramas y bigramas
    final_corpus = tokenize_text(corpus)
    
    if if_remove_stopwords:
        # Paso 2: Eliminación de stopwords
        final_corpus = remove_stopwords(final_corpus, language)
    
    # Paso 3: Eliminación de tokens con baja frecuencia
    final_corpus = remove_low_frequency_tokens(final_corpus)
    
    if if_apply_stemming:
        # Paso 4: Normalización mediante stemming
        final_corpus = apply_stemming(final_corpus)
    
    return final_corpus

import pandas as pd

def feature_extraction_execution(remove_stopwords=True, apply_stemming=True):
    """
    Función que ejecuta el flujo de procesamiento desde un archivo y guarda el resultado en un CSV.
    """
    tweet_files = [
        ("data/processed/neg/tweets.txt", "negative"),
        ("data/processed/pos/tweets.txt", "positive"),
    ]

    review_files = [
        ("data/processed/neg/neg.txt", "negative"),
        ("data/processed/pos/pos.txt", "positive"),
    ]

    general_corpus = []

    for file, sentiment in tweet_files:
        with open(file, "r", encoding="utf-8") as f:
            corpus = f.read().split("\n")[:-1]
        
        processed_corpus = feature_extraction(corpus, remove_stopwords, apply_stemming, 'english')
        general_corpus.extend([(line, sentiment) for line in processed_corpus])

    for file, sentiment in review_files:
        with open(file, "r", encoding="utf-8") as f:
            corpus = f.read().split("\n---------------\n")[:-1]
        
        processed_corpus = feature_extraction(corpus, remove_stopwords, apply_stemming)
        general_corpus.extend([(line, sentiment) for line in processed_corpus])

    # Crear un DataFrame y guardar el corpus procesado en un archivo CSV
    df = pd.DataFrame(general_corpus, columns=["Text", "Sentiment"])
    df["Text"] = df["Text"].apply(lambda x: " ".join(x))  # Convertir listas de palabras en cadenas

    # remplazar los texst con menos de 2 de longitud por "vacio"
    df["Text"] = df["Text"].apply(lambda x: "vacio" if len(x) < 2 else x)

    df.to_csv("data/processed/general_corpus.csv", index=False, encoding="utf-8")



if __name__ == '__main__':
    feature_extraction_execution()
    print("Procesamiento de texto final")
