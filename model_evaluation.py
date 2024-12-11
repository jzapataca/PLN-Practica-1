import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import pandas as pd
import os
from feature_extraction import feature_extraction_execution

def load_data(matrix_path, vectorizer_path, csv_path):
    """
    Carga los datos necesarios para entrenar el modelo.
    """
    matrix = np.load(matrix_path)
    with open(vectorizer_path, "rb") as file:
        vectorizer = pickle.load(file)
    
    df = pd.read_csv(csv_path)
    sentiments = df["Sentiment"].map({"positive": 1, "negative": 0}).values
    
    return matrix, sentiments

def split_data(matrix, sentiments, test_size=0.2, random_state=42):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    """
    return train_test_split(matrix, sentiments, test_size=test_size, random_state=random_state)

def train_logistic_regression(X_train, y_train):
    """
    Entrena un modelo de regresión logística.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_knn(X_train, y_train, n_neighbors=5):
    """
    Entrena un modelo de K-Nearest Neighbors.
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    """
    Entrena un modelo de árbol de decisión.
    """
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo y muestra métricas clave.
    """
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("Reporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    return precision, recall, f1

def save_model(model, model_path):
    """
    Guarda el modelo entrenado en un archivo.
    """
    with open(model_path, "wb") as file:
        pickle.dump(model, file)
    print(f"Modelo guardado exitosamente en {model_path}.")

def main():
    """
    Ejecuta el proceso completo de entrenamiento y evaluación de los modelos
    para todas las combinaciones de parámetros de preprocesamiento y ponderación.
    """
    # Parámetros de preprocesamiento
    parameter_combinations = [
        (True, True),
        (True, False),
        (False, True),
        (False, False)
    ]

    results = []

    for remove_stopwords, apply_stemming in parameter_combinations:
        print(f"\nEvaluando con remove_stopwords={remove_stopwords}, apply_stemming={apply_stemming}...")

        # Generar nuevos datos de ponderación
        feature_extraction_execution(remove_stopwords=remove_stopwords, apply_stemming=apply_stemming)

        # Evaluar para TF-IDF
        print("\nEvaluando TF-IDF...")
        tfidf_path = "data/processed/ponderation/tfidf_matrix.npy"
        tfidf_vectorizer_path = "data/processed/ponderation/tfidf_vectorizer.pkl"
        csv_path = "data/processed/general_corpus.csv"
        tfidf_matrix, sentiments = load_data(tfidf_path, tfidf_vectorizer_path, csv_path)
        X_train, X_test, y_train, y_test = split_data(tfidf_matrix, sentiments)

        print("\nEntrenando modelo de regresión logística con TF-IDF...")
        logistic_model = train_logistic_regression(X_train, y_train)
        precision, recall, f1 = evaluate_model(logistic_model, X_test, y_test)
        save_model(logistic_model, f"models/logistic_regression_tfidf_{remove_stopwords}_{apply_stemming}.pkl")
        results.append(("TF-IDF", "Logistic Regression", remove_stopwords, apply_stemming, precision, recall, f1))

        print("\nEntrenando modelo de K-Nearest Neighbors con TF-IDF...")
        knn_model = train_knn(X_train, y_train)
        precision, recall, f1 = evaluate_model(knn_model, X_test, y_test)
        save_model(knn_model, f"models/knn_tfidf_{remove_stopwords}_{apply_stemming}.pkl")
        results.append(("TF-IDF", "KNN", remove_stopwords, apply_stemming, precision, recall, f1))

        print("\nEntrenando modelo de árbol de decisión con TF-IDF...")
        decision_tree_model = train_decision_tree(X_train, y_train)
        precision, recall, f1 = evaluate_model(decision_tree_model, X_test, y_test)
        save_model(decision_tree_model, f"models/decision_tree_tfidf_{remove_stopwords}_{apply_stemming}.pkl")
        results.append(("TF-IDF", "Decision Tree", remove_stopwords, apply_stemming, precision, recall, f1))

        # Evaluar para TO
        print("\nEvaluando TO...")
        to_path = "data/processed/ponderation/to_matrix.npy"
        to_vectorizer_path = "data/processed/ponderation/to_vectorizer.pkl"
        to_matrix, sentiments = load_data(to_path, to_vectorizer_path, csv_path)
        X_train, X_test, y_train, y_test = split_data(to_matrix, sentiments)

        print("\nEntrenando modelo de regresión logística con TO...")
        logistic_model = train_logistic_regression(X_train, y_train)
        precision, recall, f1 = evaluate_model(logistic_model, X_test, y_test)
        save_model(logistic_model, f"models/logistic_regression_to_{remove_stopwords}_{apply_stemming}.pkl")
        results.append(("TO", "Logistic Regression", remove_stopwords, apply_stemming, precision, recall, f1))

        print("\nEntrenando modelo de K-Nearest Neighbors con TO...")
        knn_model = train_knn(X_train, y_train)
        precision, recall, f1 = evaluate_model(knn_model, X_test, y_test)
        save_model(knn_model, f"models/knn_to_{remove_stopwords}_{apply_stemming}.pkl")
        results.append(("TO", "KNN", remove_stopwords, apply_stemming, precision, recall, f1))

        print("\nEntrenando modelo de árbol de decisión con TO...")
        decision_tree_model = train_decision_tree(X_train, y_train)
        precision, recall, f1 = evaluate_model(decision_tree_model, X_test, y_test)
        save_model(decision_tree_model, f"models/decision_tree_to_{remove_stopwords}_{apply_stemming}.pkl")
        results.append(("TO", "Decision Tree", remove_stopwords, apply_stemming, precision, recall, f1))

    # Mostrar resultados
    print("\nResultados finales:")
    results_df = pd.DataFrame(results, columns=["Ponderation", "Model", "Remove Stopwords", "Apply Stemming", "Precision", "Recall", "F1-Score"])
    print(results_df)
    results_df.to_csv("models/evaluation_results.csv", index=False)

# Ejecutar el entrenamiento
if __name__ == "__main__":
    main()