# Téléchargement du modèle BART et de son tokeniseur
# from download_BART_model import download_model
# model, tokenizer = download_model()

# Importations
import os
import glob
import pandas as pd

def find_corpus_folder(directory='corpus_csv'):
    # Parcourir le répertoire spécifié (par défaut : répertoire 'corpus_csv')
    for root, dirs, files in os.walk(os.path.abspath(os.sep)):
        # Vérifier si le dossier cible est présent dans les répertoires actuels
        if directory in dirs:
            # Renvoyer le chemin complet du dossier cible s'il est trouvé
            return os.path.join(root, directory)
    # Si le dossier cible n'est pas trouvé, renvoyer None
    return None

def get_FEN_vocab():
    # Détermine le chemin du corpus
    corpus_path = find_corpus_folder(directory='corpus_csv')
    # Ajoute le motif de recherche pour tous les fichiers CSV dans le chemin du corpus
    corpus_path = corpus_path + "/*.csv"
    # Initialise le vocabulaire FEN comme un ensemble pour éviter les doublons
    FEN_vocab = set()
    # Parcours tous les fichiers CSV dans le corpus
    for csv_match_path in glob.glob(corpus_path):
        # Charge le fichier CSV dans un DataFrame pandas
        df = pd.read_csv(csv_match_path)  
        # Parcours toutes les notations FEN dans la colonne 'FEN_notation'
        for FEN in df['FEN_notation']:
            # Parcours chaque caractère dans la notation FEN
            for car in FEN:
                # Ajoute le caractère au vocabulaire FEN
                FEN_vocab.add(car)      
    # Convertit FEN_vocab en une liste
    FEN_vocab = list(FEN_vocab)
    # Ajoute "<end>" au début de la liste
    FEN_vocab.insert(0, "<end>")
    # Insère "<start>" au début de la liste
    FEN_vocab.insert(0, "<start>")
    # Renvoie FEN_vocab
    return FEN_vocab