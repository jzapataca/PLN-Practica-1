import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('models/evaluation_results.csv')

def F1ScoreMaximo(df):
    # F1-Score máximo por algoritmo
    max_f1_per_algorithm = df.groupby('Model')['F1-Score'].max()

    # Graficar le F1-Score máximo por cada algortimo
    plt.figure(figsize=(10, 10))
    max_f1_per_algorithm.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('F1-Score máximo por algoritmo', fontsize=16)
    plt.xlabel('Algoritmo', fontsize=14)
    plt.ylabel('F1-Score', fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('graficas/f1_score_maximo.png')
    plt.close()

def F1ScorePond(df):
    # F1-Score medio por ponderación y algoritmo
    avg_f1_by_ponderation = df.groupby(['Model', 'Ponderation'])['F1-Score'].mean().unstack()

    # Graficar el F1-Score medio por ponderación y algoritmo
    avg_f1_by_ponderation.plot(kind='bar', figsize=(12, 6), color=['skyblue', 'orange'], edgecolor='black')
    plt.title('F1-Score medio por ponderación y algoritmo', fontsize=16)
    plt.xlabel('Algoritmo', fontsize=14)
    plt.ylabel('F1-Score', fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.legend(title='Ponderación', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('graficas/f1_score_pond.png')
    plt.close()

def F1ScoreTechnique(df):
    # Crear una nueva columna para representar la técnica de reducción
    df['Reduction Technique'] = (
        'Stopwords=' + df['Remove Stopwords'].astype(str) + 
        ', Stemming=' + df['Apply Stemming'].astype(str)
    )

    # F1-Score medio por técnica de reducción y algoritmo
    avg_f1_by_reduction = df.groupby(['Model', 'Reduction Technique'])['F1-Score'].mean().unstack()

    # Graficar el F1-Score medio por técnica de reducción y algoritmo
    avg_f1_by_reduction.plot(kind='bar', figsize=(14, 8), edgecolor='black')
    plt.title('F1-Score medio por técnica de reducción y algoritmo', fontsize=16)
    plt.xlabel('Algoritmo', fontsize=14)
    plt.ylabel('F1-Score', fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.legend(title='Técnica de Reducción', fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('graficas/f1_score_technique.png')
    plt.close()

F1ScoreMaximo(df)
F1ScorePond(df)
F1ScoreTechnique(df)