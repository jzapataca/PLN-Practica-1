import numpy as np
from pprint import pprint

def calculate_file_metrics(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            words = text.split("\n")[:-1]
            word_counts = [len(word.split()) for word in words]
            word_counts = np.array(word_counts)

            metrics = {
                "Mean": float(np.mean(word_counts)),
                "Std": float(np.std(word_counts)),
                "Min": float(np.min(word_counts)),
                "Max": float(np.max(word_counts)),
                "25% Quartile": float(np.percentile(word_counts, 25)),
                "50% Quartile": float(np.percentile(word_counts, 50)),
                "75% Quartile": float(np.percentile(word_counts, 75)),
                "Sum": float(np.sum(word_counts))
            }
            return metrics
    except FileNotFoundError:
        print(f"Archivo no encontrado: {file_path}")
        return None

def calculate_metrics(directories):
    for directory in directories:
        print(f"\nMétricas para el directorio: {directory}")
        
        # Lista de archivos a procesar
        files_to_process = [
            f"{directory}/tweets.txt",
            f"{directory}/{'pos.txt' if 'pos' in directory else 'neg.txt'}"
        ]
        
        # Métricas por archivo individual
        all_words = []
        for file_path in files_to_process:
            print(f"\nArchivo: {file_path}")
            metrics = calculate_file_metrics(file_path)
            if metrics:
                pprint(metrics)
                
            # Recolectar palabras para métricas combinadas
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    words = file.read().split("\n")[:-1]
                    all_words.extend(words)
            except FileNotFoundError:
                continue

        # Métricas combinadas del directorio
        word_counts = [len(word.split()) for word in all_words]
        word_counts = np.array(word_counts)

        print(f"\nMétricas combinadas para {directory}:")
        metrics = {
            "Mean": float(np.mean(word_counts)),
            "Std": float(np.std(word_counts)),
            "Min": float(np.min(word_counts)),
            "Max": float(np.max(word_counts)),
            "25% Quartile": float(np.percentile(word_counts, 25)),
            "50% Quartile": float(np.percentile(word_counts, 50)),
            "75% Quartile": float(np.percentile(word_counts, 75)),
            "Sum": float(np.sum(word_counts))
        }
        pprint(metrics)
        print("-" * 50)

directories = [
    "./data/interim/pos",
    "./data/interim/neg"
]

calculate_metrics(directories)