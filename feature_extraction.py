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

def main(corpus):
    """
    Función principal que ejecuta todo el flujo de procesamiento.
    """
    # Paso 1: Tokenización en unigramas y bigramas
    tokenized_corpus = tokenize_text(corpus)
    
    # Paso 2: Eliminación de stopwords
    no_stopwords_corpus = remove_stopwords(tokenized_corpus)
    
    # Paso 3: Eliminación de tokens con baja frecuencia
    filtered_corpus = remove_low_frequency_tokens(no_stopwords_corpus)
    
    # Paso 4: Normalización mediante stemming
    normalized_corpus = apply_stemming(filtered_corpus)
    
    return normalized_corpus

if __name__ == "__main__":

    tweet_files = [
        "data/processed/neg/tweets.txt",
        "data/processed/pos/tweets.txt",
    ]

    review_files = [
        "data/processed/neg/neg.txt",
        "data/processed/pos/pos.txt",
    ]
    
    general_corpus = []

    for file in tweet_files:
        with open(file, "r", encoding="utf-8") as f:
            corpus = f.read().split("\n")[:-1]
        
        processed_corpus = main(corpus)
        general_corpus.extend(processed_corpus)

    for file in review_files:
        with open(file, "r", encoding="utf-8") as f:
            corpus = f.read().split("\n---------------\n")[:-1]
        
        processed_corpus = main(corpus)
        general_corpus.extend(processed_corpus)

    # Guardar el corpus procesado
    with open("data/processed/general_corpus.txt", "w", encoding="utf-8") as f:
        for line in general_corpus:
            f.write(" ".join(line) + "\n")