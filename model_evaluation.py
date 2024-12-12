import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import pandas as pd
from feature_extraction import feature_extraction_execution
from transformers import pipeline
from tqdm import tqdm

def load_data(matrix_path, vectorizer_path, csv_path):
    """
    Carga los datos necesarios para entrenar el modelo.
    """
    matrix = np.load(matrix_path)
    # with open(vectorizer_path, "rb") as file:
    #     vectorizer = pickle.load(file)
    
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
    scoring = ["precision", "recall", "f1"]
    
    y_pred = model.predict(X_test)    
    scores = cross_validate(model, X_test, y_test, scoring=scoring, cv=10)
    scores_df = pd.DataFrame(scores)
    
    print(f"Scores de validación cruzada: {scores_df}")
    precision = scores_df["test_precision"].mean()
    recall = scores_df["test_recall"].mean()
    f1 = scores_df["test_f1"].mean()
    
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

def load_and_sample_data(csv_path, test_size=0.2, random_state=42):
    """
    Carga los datos del CSV y selecciona una muestra del 20%
    """
    df = pd.read_csv(csv_path)
    # Usar train_test_split para obtener una muestra estratificada
    _, sample_df = train_test_split(
        df, 
        test_size=test_size,
        random_state=random_state,
        stratify=df['Sentiment']
    )
    return sample_df

def get_huggingface_predictions(texts, classifier):
    """
    Obtiene predicciones usando el modelo de HuggingFace
    """
    predictions = []
    for text in tqdm(texts, desc="Obteniendo predicciones"):
        try:
            result = classifier(text)[0]
            # Ignorar predicciones neutrales y mapear solo positivas y negativas
            if result['label'] in ['Positive', 'Negative']:
                pred = 1 if result['label'] == 'Positive' else 0
                predictions.append(pred)
            else:
                # Para predicciones neutrales, usar el score para decidir
                predictions.append(1 if result['score'] >= 0.5 else 0)
        except Exception as e:
            print(f"Error procesando texto: {e}")
            predictions.append(0)
    return np.array(predictions)

def evaluate_with_cross_validation(X, y, classifier, n_splits=10):
    """
    Evalúa usando validación cruzada con 10 folds
    """
    scores = {
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    # Implementar validación cruzada manual 
    indices = np.arange(len(X))
    np.random.shuffle(indices)  # Aleatorizar los índices
    fold_size = len(X) // n_splits
    
    for i in range(n_splits):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < n_splits-1 else len(X)
        
        # Obtener índices para este fold
        test_indices = indices[start_idx:end_idx]
        train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
        
        # Separar datos de prueba y entrenamiento
        X_test = X[test_indices]
        y_test = y[test_indices]
        
        # Obtener predicciones
        y_pred = get_huggingface_predictions(X_test, classifier)
        
        # Verificar que hay predicciones positivas y negativas
        if len(np.unique(y_pred)) > 1 and len(np.unique(y_test)) > 1:
            scores['precision'].append(precision_score(y_test, y_pred))
            scores['recall'].append(recall_score(y_test, y_pred))
            scores['f1'].append(f1_score(y_test, y_pred))
        else:
            print(f"Advertencia: Fold {i} tiene predicciones o etiquetas de una sola clase")
            continue
    
    # Calcular promedios solo si hay scores válidos
    if any(scores.values()):
        avg_scores = {k: np.mean(v) for k, v in scores.items()}
        std_scores = {k: np.std(v) for k, v in scores.items()}
    else:
        print("Error: No se pudieron calcular métricas válidas")
        avg_scores = {k: 0.0 for k in scores.keys()}
        std_scores = {k: 0.0 for k in scores.keys()}
    
    return avg_scores, std_scores

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

    # Cargar y muestrear datos
    sample_df = load_and_sample_data(csv_path)
    
    # Preparar datos
    X = sample_df['Text'].values
    y = (sample_df['Sentiment'] == 'positive').astype(int).values
    
    # Inicializar modelo de HuggingFace
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    classifier = pipeline("text-classification", model=model_name)
    
    # Evaluar usando validación cruzada
    print("Evaluando modelo con validación cruzada...")
    avg_scores, std_scores = evaluate_with_cross_validation(X, y, classifier)
    
    # Mostrar resultados
    print("\nResultados de la validación cruzada:")
    for metric in ['precision', 'recall', 'f1']:
        print(f"{metric.capitalize()}: {avg_scores[metric]:.4f} (+/- {std_scores[metric]:.4f})")
    
    # Obtener predicciones finales para toda la muestra
    print("\nObteniendo predicciones finales...")
    final_predictions = get_huggingface_predictions(X, classifier)
    
    # Mostrar reporte de clasificación completo
    print("\nReporte de clasificación completo:")
    print(classification_report(y, final_predictions))
    
    # Crear un DataFrame con los resultados del modelo de HuggingFace
    huggingface_results = {
        "Metric": ["Precision", "Recall", "F1"],
        "Score": [avg_scores["precision"], avg_scores["recall"], avg_scores["f1"]],
        "Std Dev": [std_scores["precision"], std_scores["recall"], std_scores["f1"]]
    }
    huggingface_results_df = pd.DataFrame(huggingface_results)
    
    # Guardar el DataFrame en un archivo CSV
    huggingface_results_df.to_csv("models/huggingface_results.csv", index=False)
    
    print("\nResultados del modelo de HuggingFace guardados en 'models/huggingface_results.csv'")
    

# Ejecutar el entrenamiento
if __name__ == "__main__":
    main()