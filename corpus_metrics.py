import numpy as np
from pprint import pprint
def calculate_metrics(file_paths):

    for file_path in file_paths:
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
        print(file_path)
        pprint(metrics)
        print()

file_paths = [
    "./data/interim/pos/tweets.txt",
    "./data/interim/neg/tweets.txt"
]

calculate_metrics(file_paths)